"""Detect people on every Nth frame of a whole video (Milestone 2, step 1).

The COCO-pretrained RF-DETR `person` class on whole frames: no labelling needed. Players,
referees, coaches and spectators are all detected here; telling our near-side players
apart is tracking + court position (later steps), not detection.

Writes data/cache/<id>/person_detections.npz (format: setter_eyes.detect.video), resumable.
Settings: `detect.person_*` and `models.person_detector` in config/pipeline.yaml.

Usage:
  python scripts/detect_people.py --video titan-rise_set1 [--start 60 --seconds 30] [--restart]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from setter_eyes import config
from setter_eyes.detect.tiled import canonical_names, load_detector, model_ids, predict_full
from setter_eyes.detect.video import detect_video
from setter_eyes.ingest import probe
from setter_eyes.labels import load_manifest


def main() -> int:
    pipeline = config.load("pipeline")
    paths, det_cfg = pipeline["paths"], pipeline["detect"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--checkpoint", type=Path, default=pipeline["models"]["person_detector"],
                    help="default: models.person_detector (null = COCO-pretrained)")
    ap.add_argument("--stride", type=int, default=det_cfg["person_stride"])
    ap.add_argument("--threshold", type=float, default=det_cfg["person_store_threshold"])
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float, help="only this long (default: to the end)")
    ap.add_argument("--out", type=Path, help="default: data/cache/<id>/person_detections.npz")
    ap.add_argument("--restart", action="store_true", help="ignore a previous partial run")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    out = args.out or config.repo_path(paths["cache_dir"]) / entry.key / "person_detections.npz"
    info = probe(entry.path)
    checkpoint = config.repo_path(args.checkpoint) if args.checkpoint else None
    model = load_detector(det_cfg["person_model"], checkpoint,
                          det_cfg["person_input_resolution"], fp16=True)
    keep = {i: c for i, c in canonical_names(model_ids(model, checkpoint is not None)).items()
            if c == "person"}

    def detect(bgr):
        return [(box, score) for _, box, score in
                predict_full(model, bgr, args.threshold, keep)]

    detect_video(entry.path, out, detect, stride=args.stride, threshold=args.threshold,
                 fps=info.fps, total_frames=info.frame_count or int(info.duration * info.fps),
                 first=round(args.start * info.fps),
                 last=round((args.start + args.seconds) * info.fps) if args.seconds else None,
                 save_every=det_cfg["ball_save_every"], restart=args.restart,
                 meta={"video": entry.key, "checkpoint": str(checkpoint or "coco-pretrained")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
