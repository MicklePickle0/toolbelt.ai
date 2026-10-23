"""Extract frames for labelling in an image-based tool (Roboflow, Label Studio, ...).

Files are named <key>_f<frame>.jpg, which the COCO importer maps back to frame
indices. (CVAT can take the video directly instead.)

Usage:
  python scripts/extract_frames.py VIDEO --key KEY --every 15 [--start 60 --end 120] [--out DIR]
  python scripts/extract_frames.py --video KEY --every 15     # video from data/manifest.yaml
  python scripts/extract_frames.py VIDEO --key KEY --bursts 20 --burst-len 10
      # 20 runs of 10 consecutive frames, one per equal slice of the video (motion context)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from setter_eyes import config
from setter_eyes.ingest import (
    burst_indices,
    frame_filename,
    iter_frames,
    probe,
    read_frames,
    write_image,
)
from setter_eyes.labels import load_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="video file (or use --video)")
    ap.add_argument("--video", help="manifest video key")
    ap.add_argument("--manifest", type=Path, help="default: paths.manifest_path in pipeline.yaml")
    ap.add_argument("--key", help="name prefix when giving a path (default: file stem)")
    ap.add_argument("--every", type=int, default=15, help="keep every Nth frame")
    ap.add_argument("--bursts", type=int,
                    help="instead of --every: this many runs of consecutive frames")
    ap.add_argument("--burst-len", type=int, default=10, help="frames per burst")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, help="seconds")
    ap.add_argument("--out", type=Path, help="default: <cache_dir>/<key>/frames")
    args = ap.parse_args()

    pipeline = config.load("pipeline")
    if args.video:
        manifest = args.manifest or config.repo_path(pipeline["paths"]["manifest_path"])
        entry = load_manifest(manifest).video(args.video)
        path, key = entry.path, entry.key
    elif args.path:
        path = Path(args.path)
        key = args.key or path.stem
    else:
        ap.error("give a video path or --video KEY")

    info = probe(path)
    if info.needs_normalising:
        print("WARNING: VFR/rotated video; normalise first (scripts/normalize_video.py)")
    out = args.out or config.repo_path(pipeline["paths"]["cache_dir"]) / key / "frames"
    start = round(args.start * info.fps)
    end = round(args.end * info.fps) if args.end is not None else None
    quality = pipeline["ingest"]["jpeg_quality"]

    if args.bursts:
        if end is None and info.frame_count is None:
            ap.error("frame count unknown; pass --end")
        indices = burst_indices(start, end if end is not None else info.frame_count,
                                args.bursts, args.burst_len, seed=key)
        frames = read_frames(path, indices)
    else:
        frames = iter_frames(path, start=start, end=end, step=args.every)
    n = 0
    for frame in frames:
        write_image(out / frame_filename(key, frame.index), frame.image, quality)
        n += 1
    print(f"wrote {n} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
