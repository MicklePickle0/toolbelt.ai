"""Detect the ball through a whole video with the tiled RF-DETR detector.

Samples at video.sample_hz (every 2nd frame at 59.94 fps, every frame at 30 fps), 1080p
to 4K. Two savings, both on by default:
  - only during play: frames outside data/cache/<id>/play_spans.json (scripts/find_play.py)
    are skipped (--all-frames: detect everywhere);
  - ball following: while a moving ball is followed, only a tile-sized window around it
    is detected, not every tile (setter_eyes.detect.follow; --no-follow: every tile).

Ball following against a full search, 40 s of titan-rise (2026-10-27): 2.0x faster, the
match ball found in 98.1% of the frames the full search found it in. It beat the other
version tried (one crop around the last sure ball, `--roi`, since removed): 2.4x faster
but 96.4%, missing half as many moving balls again.

Writes data/cache/<id>/ball_detections.npz (format: setter_eyes.detect.video). Saved every
`ball_save_every` frames and resumable: rerunning with the same settings continues after
the last saved frame (use --restart to start over).

Settings: `detect:` and `video:` in config/pipeline.yaml; the detector is
`models.ball_detector`.

Usage:
  python scripts/detect_ball.py --video titan-rise_set1 [--start 60 --seconds 30] [--restart]
                                [--all-frames] [--no-follow]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from setter_eyes import config
from setter_eyes.detect.follow import BallFollower
from setter_eyes.detect.tiled import (
    ball_tiling,
    canonical_names,
    load_detector,
    model_ids,
    predict_tiled,
    predict_windows,
)
from setter_eyes.detect.video import detect_video
from setter_eyes.ingest import probe
from setter_eyes.labels import load_manifest
from setter_eyes.rally.activity import in_spans


def main() -> int:
    pipeline = config.load("pipeline")
    paths, det_cfg, ds = pipeline["paths"], pipeline["detect"], pipeline["datasets"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--checkpoint", type=Path,
                    default=config.repo_path(pipeline["models"]["ball_detector"]))
    ap.add_argument("--model", default=pipeline["train"]["model"])
    ap.add_argument("--stride", type=int, help="every Nth frame (default: from video.sample_hz)")
    ap.add_argument("--threshold", type=float, default=det_cfg["ball_store_threshold"])
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float, help="only this long (default: to the end)")
    ap.add_argument("--out", type=Path, help="default: data/cache/<id>/ball_detections.npz")
    ap.add_argument("--restart", action="store_true", help="ignore a previous partial run")
    ap.add_argument("--all-frames", action="store_true",
                    help="detect between rallies too (ignore play_spans.json)")
    ap.add_argument("--no-follow", action="store_true",
                    help="every tile on every frame (no ball following)")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    cache = config.repo_path(paths["cache_dir"]) / entry.key
    out = args.out or cache / "ball_detections.npz"
    info = probe(entry.path)
    det_cfg = config.for_video(pipeline, info.width)["detect"]   # pixel settings, this video
    stride = args.stride or config.sample_stride(info.fps)

    keep, settings = None, []
    spans_path = cache / "play_spans.json"
    if det_cfg["ball_play_only"] and not args.all_frames:
        if spans_path.exists():
            spans = json.loads(spans_path.read_text(encoding="utf-8"))
            keep = in_spans([tuple(s) for s in spans["spans"]])
            settings.append(f"play_spans={spans['spans']!r}")
            print(f"play only: {spans['kept_share']:.0%} of the video ({spans_path})")
        else:
            print(f"no {spans_path}: detecting every frame (run scripts/find_play.py first "
                  "to skip the time between rallies)")
    grid, resolution = ball_tiling(det_cfg, info.width)
    overlap = ds["tile_overlap"]
    settings.append(f"tiling={grid[0]}x{grid[1]}@{resolution}")
    print(f"tiles: {grid[0]}x{grid[1]} at {resolution} px ({info.width} px wide video)")
    model = load_detector(args.model, args.checkpoint, resolution, fp16=True)
    ball = {i: c for i, c in canonical_names(model_ids(model, True)).items() if c == "ball"}

    def detect_windows(bgr, windows):
        return [(box, score) for _, box, score in
                predict_windows(model, bgr, windows, args.threshold, ball)]

    def detect_tiles(bgr, _index):
        return [(box, score) for _, box, score in
                predict_tiled(model, bgr, grid, overlap, args.threshold, ball)]

    follow = det_cfg["ball_follow"]
    follower = None
    detect = detect_tiles
    if follow["enabled"] and not args.no_follow:
        follower = BallFollower(detect_windows, grid, overlap, info.fps, follow)
        detect = follower
        settings.append(f"follow={json.dumps(follow, sort_keys=True)}")

    detect_video(entry.path, out, detect, stride=stride, threshold=args.threshold,
                 fps=info.fps, total_frames=info.frame_count or int(info.duration * info.fps),
                 first=round(args.start * info.fps),
                 last=round((args.start + args.seconds) * info.fps) if args.seconds else None,
                 save_every=det_cfg["ball_save_every"], restart=args.restart,
                 keep=keep, settings=" ".join(settings),
                 meta={"video": entry.key, "checkpoint": str(args.checkpoint)})
    if follower is not None:
        print(f"ball following: {follower.summary()} ({grid[0] * grid[1]} without it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
