"""Cut video clips from the event DB: every rally behind a stat, or every touch of an action.

Rallies credited with a stat (stats/sheet.py: kills, aces, attack_errors, ...) or every
touch of an action (attack, serve, receive, ...), optionally for one player, become short
clips (from the 1280-wide proxy when there is one, else the original), plus a single
highlight reel. Reads the labels DB or the reviewed pipeline DB.

Writes data/outputs/clips/<id>/<what>[_#N]/NN_m-ss.mp4 and <what>[_#N]_reel.mp4.

Usage:
  python scripts/clips.py --video titan-rise_set1 --stat kills
  python scripts/clips.py --video titan-rise_set1 --stat kills --player 13
  python scripts/clips.py --video mac-chrome_set1 --action receive --player 6 --db labels
  python scripts/clips.py --video titan-rise_set1 --stat aces --no-reel
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from setter_eyes import config, db
from setter_eyes.db.read import load_rallies
from setter_eyes.labels import load_manifest
from setter_eyes.labels.event_editor import clock
from setter_eyes.stats.actions import stat_actions
from setter_eyes.stats.sheet import SKIP, _decider
from setter_eyes.ui.review import rally_clip


def moments(rallies, fps: float, stat: str | None, action: str | None,
            player: int | None, before: float, after: float) -> list[tuple[float, float]]:
    """(start s, end s) of each clip."""
    out = []
    for r in rallies:
        if r.outcome in SKIP:                     # replayed / dismissed: counts for nothing
            continue
        if stat is not None:
            c, s = _decider(r)
            if s == stat and c is not None and (player is None or c.jersey == player):
                out.append((r.start_frame / fps - before, r.end_frame / fps + after))
        else:
            for c, a in stat_actions(r):
                if c.team == "us" and a == action and (player is None or c.jersey == player):
                    out.append((c.frame / fps - before, c.frame / fps + after))
    return out


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--stat", help="kills, aces, attack_errors, service_errors, "
                      "block_points, reception_errors, ball_handling_errors")
    what.add_argument("--action", help="serve, receive, set, attack, dig, block, ...")
    ap.add_argument("--player", type=int, help="jersey number")
    ap.add_argument("--db", choices=["labels", "pipeline"], default="labels")
    ap.add_argument("--before", type=float, help="seconds before (default 2 for a "
                    "rally, 3 for a touch)")
    ap.add_argument("--after", type=float, help="seconds after (default 2)")
    ap.add_argument("--no-reel", action="store_true", help="clips only, no joined reel")
    args = ap.parse_args()

    conn = db.connect(config.repo_path(
        paths["labels_db_path" if args.db == "labels" else "db_path"]))
    row = conn.execute("SELECT fps FROM games WHERE video_key = ?", (args.video,)).fetchone()
    if row is None:
        raise SystemExit(f"{args.video} isn't in the {args.db} DB")
    fps = row[0]
    before = args.before if args.before is not None else (2.0 if args.stat else 3.0)
    after = args.after if args.after is not None else 2.0
    spans = moments(load_rallies(conn, args.video), fps, args.stat, args.action, args.player,
                    before, after)
    name = (args.stat or args.action) + (f"_#{args.player}" if args.player else "")
    if not spans:
        print(f"no {name} in {args.video} ({args.db} DB)")
        return 0
    entry = load_manifest(config.repo_path(paths["manifest_path"])).video(args.video)
    proxy = config.repo_path(paths["cache_dir"]) / entry.key / "proxy.mp4"
    src = proxy if proxy.exists() else entry.path
    out_dir = config.repo_path(paths["output_dir"]) / "clips" / entry.key / name
    clips: list[Path] = []
    for n, (t0, t1) in enumerate(spans, start=1):
        t0 = max(0.0, t0)
        out = out_dir / f"{n:02d}_{clock(t0).replace(':', '-').split('.')[0]}.mp4"
        clips.append(rally_clip(src, out, t0, t1))
        print(f"  {out.name}: {clock(t0)} - {clock(t1)}")
    print(f"{len(clips)} clips -> {out_dir}")
    if not args.no_reel:
        listing = out_dir / "reel.txt"
        listing.write_text("".join(f"file '{c.name}'\n" for c in clips), encoding="utf-8")
        reel = out_dir.parent / f"{name}_reel.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                        str(listing), "-c", "copy", str(reel)], check=True)
        print(f"reel -> {reel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
