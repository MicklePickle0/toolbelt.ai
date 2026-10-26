"""Detect the ball on every Nth frame of a whole video with the tiled RF-DETR detector.

Writes data/cache/<id>/ball_detections.npz (format: setter_eyes.detect.video). Saved every
`ball_save_every` frames and resumable: rerunning continues after the last saved frame
(use --restart to start over).

Settings: `detect:` in config/pipeline.yaml; the detector is `models.ball_detector`.

--roi searches near the ball once it is found instead of everywhere (detect/roi.py;
simulated ~2.6x fewer detector runs for about the same accuracy). Try it first on a short
stretch with --out, against a full search of the same stretch.

Usage:
  python scripts/detect_ball.py --video titan-rise_set1 [--start 60 --seconds 30] [--restart]
  python scripts/detect_ball.py --video titan-rise_set1 --roi --start 60 --seconds 60 \
      --out data/outputs/roi_check.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from setter_eyes import config
from setter_eyes.datasets.tiles import parse_grid
from setter_eyes.detect.roi import RoiPlanner
from setter_eyes.detect.tiled import (
    canonical_names,
    load_detector,
    model_ids,
    predict_crop,
    predict_tiled,
)
from setter_eyes.detect.video import detect_video
from setter_eyes.ingest import probe
from setter_eyes.labels import load_manifest


def main() -> int:
    pipeline = config.load("pipeline")
    paths, det_cfg, ds = pipeline["paths"], pipeline["detect"], pipeline["datasets"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--checkpoint", type=Path,
                    default=config.repo_path(pipeline["models"]["ball_detector"]))
    ap.add_argument("--model", default=pipeline["train"]["model"])
    ap.add_argument("--stride", type=int, default=det_cfg["ball_stride"])
    ap.add_argument("--threshold", type=float, default=det_cfg["ball_store_threshold"])
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float, help="only this long (default: to the end)")
    ap.add_argument("--out", type=Path, help="default: data/cache/<id>/ball_detections.npz")
    ap.add_argument("--restart", action="store_true", help="ignore a previous partial run")
    ap.add_argument("--roi", action="store_true", help="search near the ball (faster)")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    out = args.out or config.repo_path(paths["cache_dir"]) / entry.key / "ball_detections.npz"
    info = probe(entry.path)
    model = load_detector(args.model, args.checkpoint, det_cfg["ball_input_resolution"],
                          fp16=True)
    keep = {i: c for i, c in canonical_names(model_ids(model, True)).items() if c == "ball"}
    grid, overlap = parse_grid(ds["tiles"]), ds["tile_overlap"]

    def detect(bgr):
        return [(box, score) for _, box, score in
                predict_tiled(model, bgr, grid, overlap, args.threshold, keep)]

    def detect_crop(bgr, crop):
        return [(box, score) for _, box, score in
                predict_crop(model, bgr, crop, args.threshold, keep)]

    planner = RoiPlanner(info.width, info.height, grid, overlap, det_cfg["roi"]) \
        if args.roi else None

    detect_video(entry.path, out, detect, stride=args.stride, threshold=args.threshold,
                 fps=info.fps, total_frames=info.frame_count or int(info.duration * info.fps),
                 first=round(args.start * info.fps),
                 last=round((args.start + args.seconds) * info.fps) if args.seconds else None,
                 save_every=det_cfg["ball_save_every"], restart=args.restart,
                 meta={"video": entry.key, "checkpoint": str(args.checkpoint),
                       "roi": bool(args.roi)},
                 planner=planner, detect_crop=detect_crop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
