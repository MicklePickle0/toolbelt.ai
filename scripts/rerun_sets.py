"""Re-run every processed set with the current pipeline, safely (roadmap E6; was A17/A4/C9).

For each video that has the detection caches (on the desktop), from cached detections:
  pose_people (only when data/cache/<id>/pose.npz is missing; the one GPU step),
  find_rallies, check_blocks, find_rallies (as the app's processing steps).
Before anything, the event DB is backed up (data/setter_eyes.db.backup_<date>_before_rerun),
and per video the reviewed rallies and their touches are counted; after each video the
counts must match (reviewed rallies are kept by db/pipeline.py, never rebuilt). Any
mismatch stops the run and says how to restore the backup.

Reports per video, before and after: our touches with a located toucher (contacts.player_x)
and a named player, and the set checks (rally/sanity.py). Ends with a table to paste into
docs/todo.md.

Usage (desktop, repo root):
  python scripts/rerun_sets.py --dry-run          # what would run, nothing changed
  python scripts/rerun_sets.py                    # every processed video
  python scripts/rerun_sets.py --video mac-chrome_set1 --video titan-rise_set1
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from setter_eyes import config, match
from setter_eyes import folders as fold
from setter_eyes.labels import load_manifest
from setter_eyes.rally.sanity import check_set, load_set

STEPS = ["find_rallies", "check_blocks", "find_rallies"]
NEEDED = ["ball_path.npz", "ball_arcs.json", "people_tracks.npz"]   # cache files the steps read


def _ro(path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def snapshot(db_path: Path, key: str, cfg: dict, formats: dict, folders: dict) -> dict | None:
    """Reviewed rallies / touches, coverage of our touches, set checks, for one video."""
    if not db_path.exists():
        return None
    c = _ro(db_path)
    try:
        row = c.execute("SELECT s.id FROM sets s JOIN games g ON s.game_id = g.id "
                        "WHERE g.video_key = ?", (key,)).fetchone()
        if row is None:
            return None
        set_id = row[0]
        rev = c.execute("SELECT COUNT(*), COALESCE(SUM((SELECT COUNT(*) FROM contacts x "
                        "WHERE x.rally_id = r.id)), 0) FROM rallies r WHERE r.set_id = ? "
                        "AND r.reviewed = 1", (set_id,)).fetchone()
        cov = c.execute("SELECT COUNT(*), SUM(c.player_x IS NOT NULL), "
                        "SUM(c.player_id IS NOT NULL), SUM(c.track_id IS NOT NULL) "
                        "FROM contacts c JOIN rallies r ON c.rally_id = r.id "
                        "WHERE r.set_id = ? AND c.team = 'us'", (set_id,)).fetchone()
        game = fold.folder_of(key, folders)["game"]
        res = check_set(load_set(c, set_id), match.format_for(game, formats, cfg),
                        match.set_number(key, formats), cfg)
    finally:
        c.close()
    return {"reviewed_rallies": rev[0], "reviewed_touches": rev[1], "ours": cov[0] or 0,
            "located": cov[1] or 0, "named": cov[2] or 0, "tracked": cov[3] or 0,
            "status": res["status"], "score": "-".join(map(str, res["score"])),
            "flagged": len(res["flagged"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", action="append", help="only these videos")
    ap.add_argument("--dry-run", action="store_true", help="say what would run; change nothing")
    args = ap.parse_args()
    cfg = config.load("pipeline")
    paths = cfg["paths"]
    db_path = config.repo_path(paths["db_path"])
    cache = config.repo_path(paths["cache_dir"])
    formats = match.load(config.repo_path(paths["match_formats_path"]))
    folders = fold.load(config.repo_path(paths["folders_path"]))
    manifest = load_manifest(config.repo_path(paths["manifest_path"]))
    keys = args.video or [v.key for v in manifest.videos]

    plan = []
    for key in keys:
        missing = [f for f in NEEDED if not (cache / key / f).exists()]
        if missing:
            print(f"{key}: skipped, no caches here ({', '.join(missing)} missing)")
            continue
        steps = (["pose_people"] if not (cache / key / "pose.npz").exists() else []) + STEPS
        plan.append((key, steps))
        print(f"{key}: {' -> '.join(steps)}")
    if not plan:
        print("nothing to re-run on this machine (the caches are on the desktop)")
        return 0
    if args.dry_run:
        return 0

    backup = db_path.with_name(f"{db_path.name}.backup_{time.strftime('%Y-%m-%d')}_before_rerun")
    if db_path.exists() and not backup.exists():
        shutil.copy2(db_path, backup)
        print(f"backup: {backup}")
    rows = []
    for key, steps in plan:
        before = snapshot(db_path, key, cfg, formats, folders)
        for step in steps:
            print(f"\n== {key}: {step}", flush=True)
            r = subprocess.run([sys.executable, str(config.repo_path(f"scripts/{step}.py")),
                                "--video", key])
            if r.returncode != 0:
                print(f"{key}: {step} failed (exit {r.returncode}); next video")
                break
        after = snapshot(db_path, key, cfg, formats, folders)
        if before and after and (before["reviewed_rallies"], before["reviewed_touches"]) != \
                (after["reviewed_rallies"], after["reviewed_touches"]):
            print(f"\nSTOP: {key} had {before['reviewed_rallies']} reviewed rallies / "
                  f"{before['reviewed_touches']} touches, now {after['reviewed_rallies']} / "
                  f"{after['reviewed_touches']}. Restore the DB: copy {backup} over {db_path}.")
            return 1
        rows.append((key, before, after))

    def cov(s):
        if not s:
            return "-"
        n = s["ours"] or 1
        return (f"located {s['located']}/{s['ours']} ({100 * s['located'] / n:.0f}%), named "
                f"{s['named']} ({100 * s['named'] / n:.0f}%), tracked {s['tracked']}")
    print("\n| video | before | after | set checks after |\n|---|---|---|---|")
    for key, b, a in rows:
        print(f"| {key} | {cov(b)} | {cov(a)} | "
              f"{a['status']} {a['score']}, {a['flagged']} flagged |" if a else
              f"| {key} | {cov(b)} | - | - |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
