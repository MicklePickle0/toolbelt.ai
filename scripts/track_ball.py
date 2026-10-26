"""Track the match ball through a video and fit its flights in 3D.

Needs data/cache/<id>/ball_detections.npz (scripts/detect_ball.py) and the camera track
data/calibration/<id>_track.npz (scripts/calibrate.py + track_camera.py). Writes
  data/cache/<id>/ball_path.npz    the match ball per processed frame (pixels, measured
                                   or bridged, detection score and size)
  data/cache/<id>/ball_arcs.json   one 3D ballistic arc per flight, with confidence
Settings: `ball:` in config/pipeline.yaml. Check by eye with scripts/render_overlay.py --ball.

Usage:
  python scripts/track_ball.py --video titan-rise_set1 [--detections PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from setter_eyes import config
from setter_eyes.ball.arcs import fit_flights
from setter_eyes.ball.tracker import Detections, track_ball
from setter_eyes.calibration.track import PoseTrack
from setter_eyes.labels import load_manifest


def main() -> int:
    pipeline = config.load("pipeline")
    paths = pipeline["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--detections", type=Path)
    ap.add_argument("--out-dir", type=Path, help="default: data/cache/<id>/")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    cache = config.repo_path(paths["cache_dir"]) / entry.key
    out_dir = args.out_dir or cache
    track_path = config.repo_path(paths["calibration_dir"]) / f"{entry.key}_track.npz"
    if not track_path.exists():
        raise SystemExit(f"no camera track at {track_path}; run scripts/track_camera.py")
    cams = PoseTrack.load(track_path)
    dets = Detections.load(args.detections or cache / "ball_detections.npz")

    t0 = time.time()
    ref = cams.camera(cams.ref_frame)
    cfg = config.scale_px(pipeline["ball"], ref.width)    # pixel settings for this video
    path = track_ball(dets, cams.camera, ref, cfg)
    times = path.frames / dets.fps
    valid = path.measured & np.isfinite(path.uv).all(axis=1)
    cameras = [cams.camera(int(f)) for f in path.frames]
    arcs = fit_flights(times, path.uv_ref, path.uv, valid, cameras, path.size_px, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    path.save(out_dir / "ball_path.npz")
    good = [a for a in arcs if a.confidence >= cfg["arcs"]["min_confidence"]]
    summary = {
        "video": entry.key, "processed_frames": int(len(path.frames)),
        "ball_measured": float(path.measured.mean()),
        "ball_bridged": float((np.isfinite(path.uv).all(axis=1) & ~path.measured).mean()),
        "arcs": len(arcs), "arcs_confident": len(good),
        "median_arc_rms_px": float(np.median([a.rms_px for a in arcs])) if arcs else None,
        "median_size_ratio": float(np.nanmedian([a.size_ratio for a in good])) if good else None,
    }
    (out_dir / "ball_arcs.json").write_text(json.dumps(
        {"summary": summary, "gravity": cfg["gravity"], "fps": dets.fps, "width": ref.width,
         "arcs": [a.to_dict() for a in arcs]}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"took {time.time() - t0:.0f} s -> {out_dir / 'ball_path.npz'}, "
          f"{out_dir / 'ball_arcs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
