"""Check the rotation state machine against labelled sets (no video needed).

For each video's events.csv: the lineup implied by who served our serves (a suggestion
for the manifest's `lineups`), then the rotation followed from the serving-team sequence
alone, with the labelled servers held back, scored against those servers. With the
manifest's lineup when it has one, else the suggested one.

Usage:
  python scripts/check_rotation.py --video titan-rise_set1 --video mac-chrome_set1
"""

from __future__ import annotations

import argparse
import sqlite3

from setter_eyes import config
from setter_eyes.identity.rotation import follow_rotation, suggest_lineup
from setter_eyes.labels import load_manifest
from setter_eyes.labels.events import parse_events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default="data/manifest.yaml")
    ap.add_argument("--video", action="append", help="manifest id (default: all with labels)")
    ap.add_argument("--fps", type=float, help="video fps (default: from the event DB)")
    args = ap.parse_args()

    cfg = config.load("pipeline")
    manifest = load_manifest(args.manifest)
    db = sqlite3.connect(config.repo_path(cfg["paths"]["db_path"]))
    keys = args.video or [v.key for v in manifest.videos if v.events is not None]
    for key in keys:
        entry = manifest.video(key)
        if entry.events is None:
            print(f"== {key}: no events.csv")
            continue
        fps = args.fps or (db.execute("SELECT fps FROM games WHERE video_key = ?", (key,))
                           .fetchone() or [None])[0]
        if fps is None:
            print(f"== {key}: fps unknown (not in the event DB); pass --fps")
            continue
        rallies, _ = parse_events(entry.events, fps)
        for set_no in sorted({r.set_number for r in rallies}):
            in_set = [r for r in rallies if r.set_number == set_no and not r.replayed]
            teams = [r.serving_team for r in in_set]
            servers = [next((c.jersey for c in r.contacts if c.action == "serve"), None)
                       if r.serving_team == "us" else None for r in in_set]
            suggested, notes = suggest_lineup(teams, servers)
            print(f"== {key} set {set_no}: {len(in_set)} rallies, "
                  f"{sum(t == 'us' for t in teams)} our serves")
            print(f"  suggested lineup (from our servers): {suggested}")
            for n in notes:
                print(f"    {n}")
            lineup = entry.lineups.get(set_no)
            print(f"  manifest lineup: {lineup}")
            lineup = lineup or (suggested if None not in suggested else None)
            if lineup is None:
                print("  not every slot served; can't follow the rotation")
                continue
            states = follow_rotation(lineup, teams, cfg)
            right = wrong = 0
            for i, (st, jersey) in enumerate(zip(states, servers, strict=True), start=1):
                if jersey is None:
                    continue
                if st.server == jersey:
                    right += 1
                else:
                    wrong += 1
                    print(f"    rally {i}: rotation {st.rotation} predicts #{st.server}, "
                          f"labelled #{jersey}")
            print(f"  servers from the serving sequence alone: {right}/{right + wrong} right")


if __name__ == "__main__":
    main()
