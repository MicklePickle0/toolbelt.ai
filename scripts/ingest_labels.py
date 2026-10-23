"""Validate labelled footage from the manifest and import it into the labels DB.

Every problem in every file is reported. A video with errors is not imported
(warnings are fine). Re-running replaces that video's previous import.

Usage:
  python scripts/ingest_labels.py                  # all videos in data/manifest.yaml
  python scripts/ingest_labels.py --video KEY      # one video
  python scripts/ingest_labels.py --check          # validate only, write nothing
  options: --manifest PATH  --db PATH  --count-frames
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from setter_eyes import config, db
from setter_eyes.datasets import assign_splits
from setter_eyes.labels import LabelError, import_video, load_manifest, load_video_labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths = config.load("pipeline")["paths"]
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--db", type=Path, default=config.repo_path(paths["labels_db_path"]))
    ap.add_argument("--video", action="append", help="only this key (repeatable)")
    ap.add_argument("--check", action="store_true", help="validate only")
    ap.add_argument("--count-frames", action="store_true",
                    help="decode videos for exact frame counts (slow; for containers without one)")
    args = ap.parse_args()

    try:
        manifest = load_manifest(args.manifest)
    except FileNotFoundError:
        print(f"no manifest at {args.manifest}; copy config/manifest.example.yaml there")
        return 1
    except LabelError as e:
        print(e)
        return 1
    splits = assign_splits(manifest)
    entries = [manifest.video(k) for k in args.video] if args.video else manifest.videos

    conn = None if args.check else db.connect(args.db)
    failed = 0
    for entry in entries:
        print(f"\n== {entry.key}  [{splits[entry.key]}]  {entry.path}")
        vl = load_video_labels(manifest, entry, count_frames=args.count_frames)
        for issue in vl.issues:
            print(f"  {issue}")
        if vl.info is None or not vl.ok:
            failed += 1
            print("  NOT IMPORTED (fix the errors above)")
            continue
        info = vl.info
        print(f"  video: {info.width}x{info.height} @ {info.fps:.3f} fps, "
              f"{info.frame_count} frames")
        _summarise(vl)
        if conn is not None:
            s = import_video(conn, manifest, entry, info, vl.rallies, vl.boxes)
            print(f"  imported: game {s.game_id}, {s.sets} sets, {s.rallies} rallies, "
                  f"{s.contacts} contacts, {s.ball_samples} ball samples -> {args.db}")
    if conn is not None:
        conn.close()
    print(f"\n{len(entries) - failed}/{len(entries)} videos OK")
    return 1 if failed else 0


def _summarise(vl) -> None:
    if vl.rallies:
        contacts = [c for r in vl.rallies for c in r.contacts]
        actions = Counter(f"{c.team}:{c.action}" for c in contacts)
        winners = Counter(r.point_winner for r in vl.rallies)
        passes = Counter(c.pass_score for c in contacts if c.pass_score is not None)
        print(f"  events: {len(vl.rallies)} rallies (us {winners['us']} / them {winners['them']}),"
              f" {len(contacts)} contacts")
        print("    actions: " + ", ".join(f"{k} {v}" for k, v in sorted(actions.items())))
        if passes:
            print("    pass scores: " + ", ".join(f"{k}: {passes[k]}" for k in sorted(passes)))
    if vl.boxes is not None:
        b = vl.boxes
        empty = sum(not boxes for boxes in b.frames.values())
        print(f"  boxes: {len(b.frames)} frames ({empty} with no boxes), "
              f"ball {b.count('ball')}, person {b.count('person')}")


if __name__ == "__main__":
    raise SystemExit(main())
