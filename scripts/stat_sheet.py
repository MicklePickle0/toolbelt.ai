"""Stat sheet for our team, from an event DB (Milestone 9, first version).

Reads rallies with a point winner: your labels (--db labels, data/labels.db, filled by
ingest_labels.py) or the pipeline DB after review (--db pipeline, data/setter_eyes.db).
Prints the team sheet and a per-player table and writes
data/outputs/stats/<id>_<db>.csv. Counting rules: setter_eyes/stats/sheet.py.

Usage:
  python scripts/stat_sheet.py --video titan-rise_set1 --db labels
  python scripts/stat_sheet.py --all --db pipeline
"""

from __future__ import annotations

import argparse
import csv

from setter_eyes import config, db
from setter_eyes.db.read import load_rallies, video_keys
from setter_eyes.stats.sheet import stat_rows

COLUMNS = ["serves", "aces", "service_errors", "attacks", "kills", "attack_errors",
           "hitting_eff", "receptions", "pass_n", "pass_avg", "reception_errors", "digs",
           "free_ball_passes", "sets", "blocks", "block_points", "ball_handling_errors"]
SHORT = {"serves": "Srv", "aces": "Ace", "service_errors": "SE", "attacks": "Att",
         "kills": "K", "attack_errors": "AE", "hitting_eff": "Eff", "receptions": "Rec",
         "pass_n": "PassN", "pass_avg": "PassAvg", "reception_errors": "RE", "digs": "Dig",
         "free_ball_passes": "FBP", "sets": "Set", "blocks": "Blk",
         "block_points": "BlkPt", "ball_handling_errors": "BHE"}


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="manifest key")
    ap.add_argument("--all", action="store_true", help="every video in the DB")
    ap.add_argument("--db", choices=["labels", "pipeline"], default="labels")
    args = ap.parse_args()
    if not args.video and not args.all:
        ap.error("--video or --all")
    db_path = config.repo_path(paths["labels_db_path" if args.db == "labels" else "db_path"])
    conn = db.connect(db_path)
    scale = config.load("pass_scale")
    out_dir = config.repo_path(paths["output_dir"]) / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ([args.video] if args.video else video_keys(conn)):
        rallies = load_rallies(conn, key)
        if not rallies:
            print(f"{key}: no rallies with a point winner in {db_path.name}")
            continue
        rows = stat_rows(rallies, scale)
        t = rows["team"]
        print(f"\n== {key} ({args.db}): {t['points_us']}-{t['points_them']}, side-out "
              f"{t['side_out_won']}/{t['side_out_chances']}, on our serve "
              f"{t['serve_points_won']}/{t['serve_points_played']}")
        players = sorted((k for k in rows if k not in ("team", None)), key=int)
        order = ["team", *players] + ([None] if None in rows else [])
        print(f"{'':8s}" + "".join(f"{SHORT[c]:>8s}" for c in COLUMNS))
        for k in order:
            name = "TEAM" if k == "team" else "#?" if k is None else f"#{k}"
            print(f"{name:8s}" + "".join(f"{_fmt(rows[k].get(c)):>8s}" for c in COLUMNS))
        out = out_dir / f"{key}_{args.db}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["player", *COLUMNS])
            for k in order:
                w.writerow(["team" if k == "team" else k, *[rows[k].get(c, "") for c in COLUMNS]])
        print(f"-> {out}")
    return 0


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


if __name__ == "__main__":
    raise SystemExit(main())
