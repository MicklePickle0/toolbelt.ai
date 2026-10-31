"""Write a video's reviewed rallies (event DB) as events.csv labels.

Rallies reviewed in the web app with every touch confirmed become ordinary event labels
(docs/labelling.md), so eval_contacts.py, ingest_labels.py and train_touch_scorer.py use
them like hand labels. Rallies dismissed as not a rally are left out (no rally there);
rallies whose touches weren't all confirmed are skipped and counted.

Never overwrites: if the output file exists, it stops (pass --out to write elsewhere).

Usage:
  python scripts/export_reviewed.py --video pakmen-black_set2 [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from setter_eyes import config, db
from setter_eyes.db.read import reviewed_for_training
from setter_eyes.labels.event_editor import COLUMNS


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--out", type=Path, help="default: data/labels/<id>/events.csv")
    args = ap.parse_args()

    out = args.out or config.repo_path("data/labels") / args.video / "events.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it (use --out for another file)")
    conn = db.connect(config.repo_path(paths["db_path"]))
    rallies = [r for r in reviewed_for_training(conn, args.video) if r.outcome != "not_a_rally"]
    if not rallies:
        raise SystemExit(f"{args.video}: no reviewed rallies with every touch confirmed")
    total = conn.execute("SELECT COUNT(*) FROM rallies r JOIN sets s ON r.set_id = s.id "
                         "JOIN games g ON s.game_id = g.id WHERE g.video_key = ?",
                         (args.video,)).fetchone()[0]

    rows = []
    for r in rallies:
        base = {"set": r.set_number}
        rows.append({**base, "frame": r.start_frame, "type": "rally_start",
                     "rally": r.rally_number})
        for c in r.contacts:
            rows.append({**base, "frame": c.frame, "type": "contact", "team": c.team,
                         "jersey": c.jersey if c.jersey is not None else "",
                         "action": c.action or "",
                         "pass_score": c.pass_score if c.pass_score is not None else ""})
        rows.append({**base, "frame": r.end_frame, "type": "rally_end",
                     "point_winner": r.point_winner or "", "outcome": r.outcome or ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[*COLUMNS[:4], "rally", *COLUMNS[4:]],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{args.video}: {len(rallies)} of {total} rallies "
          f"({sum(len(r.contacts) for r in rallies)} touches) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
