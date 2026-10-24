"""Detect the ball on every Nth frame of a whole video with the tiled RF-DETR detector.

Writes data/cache/<id>/ball_detections.npz:
  frames      every processed frame index (including frames with no detection)
  det_frame   frame index of each detection, det_box (x1, y1, x2, y2) px, det_score
plus the stride, threshold and checkpoint used. Saved every `ball_save_every` frames and
resumable: rerunning continues after the last saved frame (use --restart to start over).

Settings: `detect:` in config/pipeline.yaml; the detector is `models.ball_detector`.

Usage:
  python scripts/detect_ball.py --video titan-rise_set1 [--start 60 --seconds 30] [--restart]
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np

from setter_eyes import config
from setter_eyes.datasets.tiles import parse_grid
from setter_eyes.detect.tiled import canonical_names, load_detector, model_ids, predict_tiled
from setter_eyes.ingest import iter_frames, probe
from setter_eyes.labels import load_manifest


def save(path: Path, frames, det_frame, det_box, det_score, meta: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, frames=np.array(frames, np.int64),
                        det_frame=np.array(det_frame, np.int64),
                        det_box=np.array(det_box, np.float32).reshape(-1, 4),
                        det_score=np.array(det_score, np.float32), **meta)
    tmp.replace(path)                     # never leave a half-written file behind


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
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    out = args.out or config.repo_path(paths["cache_dir"]) / entry.key / "ball_detections.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    info = probe(entry.path)
    first = round(args.start * info.fps)
    last = round((args.start + args.seconds) * info.fps) if args.seconds else None

    frames, det_frame, det_box, det_score = [], [], [], []
    if out.exists() and not args.restart:
        prev = np.load(out)
        if int(prev["stride"]) == args.stride and float(prev["threshold"]) == args.threshold:
            frames, det_frame = list(prev["frames"]), list(prev["det_frame"])
            det_box, det_score = list(prev["det_box"]), list(prev["det_score"])
            if frames:
                first = max(first, int(frames[-1]) + 1)
                print(f"resuming after frame {frames[-1]} ({len(frames)} frames done)")
    # Keep the stride grid aligned with the original start when resuming.
    first += (-first) % args.stride
    meta = {"video": entry.key, "stride": args.stride, "threshold": args.threshold,
            "checkpoint": str(args.checkpoint), "fps": info.fps}

    model = load_detector(args.model, args.checkpoint, det_cfg["ball_input_resolution"],
                          fp16=True)
    keep = {i: c for i, c in canonical_names(model_ids(model, True)).items() if c == "ball"}
    grid, overlap = parse_grid(ds["tiles"]), ds["tile_overlap"]

    # Decode on a background thread so the GPU never waits for the next frame.
    q: queue.Queue = queue.Queue(maxsize=8)

    def reader():
        for frame in iter_frames(entry.path, start=first, end=last, step=args.stride):
            q.put(frame)
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    t0, n = time.time(), 0
    total = ((last or info.frame_count or int(info.duration * info.fps)) - first) // args.stride
    while (frame := q.get()) is not None:
        for _, box, score in predict_tiled(model, frame.image, grid, overlap, args.threshold,
                                           keep):
            det_frame.append(frame.index)
            det_box.append(box)
            det_score.append(score)
        frames.append(frame.index)
        n += 1
        if n % det_cfg["ball_save_every"] == 0:
            save(out, frames, det_frame, det_box, det_score, meta)
            rate = n / (time.time() - t0)
            print(f"  frame {frame.index}: {n}/{total} at {rate:.1f}/s, "
                  f"~{(total - n) / rate / 60:.0f} min left", flush=True)
    save(out, frames, det_frame, det_box, det_score, meta)
    print(f"done: {len(frames)} frames, {len(det_frame)} detections "
          f"({n / max(1e-9, time.time() - t0):.1f} frames/s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
