"""Print a video's metadata and whether it should be normalised before labelling.

Usage: python scripts/probe_video.py VIDEO [--count-frames]
"""

from __future__ import annotations

import argparse
import json

from setter_eyes.ingest import probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--count-frames", action="store_true",
                    help="decode the whole file for an exact frame count (slow)")
    args = ap.parse_args()

    info = probe(args.video, count_frames=args.count_frames)
    print(json.dumps(info.to_dict(), indent=2))
    if info.needs_normalising:
        print("\nWARNING: variable frame rate or rotation metadata. Normalise before labelling:\n"
              f"  python scripts/normalize_video.py {args.video} OUT.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
