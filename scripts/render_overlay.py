"""Render an overlay clip from saved results (no recomputation).

Draws the calibrated court, net and antennas on every frame of [--start, --start +
--seconds), using data/calibration/<id>_track.npz (scripts/track_camera.py), with frame,
pan angle and tracking confidence in the corner. With --ball, also the tracked ball from
data/cache/<id>/ball_* (scripts/detect_ball.py, track_ball.py): grey boxes = raw
detections, green circle = detected ball, orange = bridged gap, cyan = 1 s trail, white
curve = fitted 3D flight (grey = low confidence). --ground adds the drop line to the
floor and the ball's ground trail. Encoded as H.264 so any player opens it.

Usage:
  python scripts/render_overlay.py --video titan-rise_set1 [--start 60 --seconds 60 --width 1920]
  python scripts/render_overlay.py --video titan-rise_set1 --ball --ground --start 60
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from setter_eyes import config
from setter_eyes.calibration import court_model, draw_court
from setter_eyes.calibration.track import PoseTrack, _angle_deg
from setter_eyes.ingest import iter_frames_scaled, probe
from setter_eyes.labels import load_manifest


def main() -> int:
    pipeline = config.load("pipeline")
    paths = pipeline["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--start", type=float, default=60.0, help="seconds")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--out", type=Path, help="default: data/outputs/calibration/<id>_demo.mp4,"
                    " or data/outputs/ball/<id>_ball.mp4 with --ball")
    ap.add_argument("--ball", action="store_true", help="draw the tracked ball and its arcs")
    ap.add_argument("--ground", action="store_true", help="with --ball: drop line + ground trail")
    ap.add_argument("--ball-dir", type=Path, help="default: data/cache/<id>/")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")
    entry = load_manifest(args.manifest).video(args.video)
    track_path = config.repo_path(paths["calibration_dir"]) / f"{entry.key}_track.npz"
    if not track_path.exists():
        raise SystemExit(f"no track at {track_path}; run scripts/track_camera.py first")
    track = PoseTrack.load(track_path)
    model = court_model(entry.net_height_m)
    fps = probe(entry.path).fps
    f0, f1 = round(args.start * fps), round((args.start + args.seconds) * fps)
    s = args.width / track.base.width
    height = round(track.base.height * s / 2) * 2
    ref_R = cv2.Rodrigues(track.rvecs[np.searchsorted(track.frames, track.ref_frame)])[0]

    ball = None
    if args.ball:
        from setter_eyes.ball.overlay import BallOverlay

        ball = BallOverlay(args.ball_dir or config.repo_path(paths["cache_dir"]) / entry.key,
                           pipeline["ball"]["arcs"]["min_confidence"])
    out = args.out or (
        config.repo_path(paths["output_dir"]) / "ball" / f"{entry.key}_ball.mp4" if ball
        else config.repo_path(paths["output_dir"]) / "calibration" / f"{entry.key}_demo.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{args.width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         str(out)], stdin=subprocess.PIPE)
    n = 0
    try:
        for frame in iter_frames_scaled(entry.path, args.width):
            if frame.index < f0:
                continue
            if frame.index >= f1:
                break
            cam = track.camera(frame.index)
            i = int(np.searchsorted(track.frames, frame.index))
            pan = _angle_deg(cam.R, ref_R)
            label = (f"{entry.key}  f{frame.index}  pan {pan:.2f} deg  "
                     f"track conf {track.confidence[i]:.2f}")
            img = draw_court(frame.image, cam.scaled(s), model, None if ball else label,
                             keypoints=False)
            if ball is not None:
                extra = ball.draw(img, frame.index, cam.scaled(s), s, ground=args.ground)
                _text(img, f"f{frame.index}   {extra}")
            enc.stdin.write(img.tobytes())
            n += 1
    finally:
        enc.stdin.close()
        enc.wait()
    print(f"wrote {n} frames ({n / fps:.0f} s) -> {out}")
    return 0


def _text(image: np.ndarray, label: str) -> None:
    t = max(1, round(image.shape[1] / 960))
    for colour, width in (((0, 0, 0), 4 * t), ((255, 255, 255), t)):
        cv2.putText(image, label, (10 * t, 30 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, colour,
                    width, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
