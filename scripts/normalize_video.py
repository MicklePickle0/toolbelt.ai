"""Re-encode a video to constant frame rate H.264 (rotation baked in, audio dropped).

Do this BEFORE labelling a phone/VFR recording, then label and register the
normalised file: frame indices are the shared key between labels and pipeline.

Usage: python scripts/normalize_video.py SRC DST [--fps 59.94] [--crf 16]
"""

from __future__ import annotations

import argparse
import json

from setter_eyes.ingest import normalise, probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--fps", type=float, help="target fps (default: source average)")
    ap.add_argument("--crf", type=int, default=16, help="x264 quality, lower = better")
    args = ap.parse_args()

    out = normalise(args.src, args.dst, fps=args.fps, crf=args.crf)
    print(json.dumps(probe(out, count_frames=True).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
