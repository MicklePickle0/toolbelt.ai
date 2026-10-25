"""Render an overlay clip from saved results (no recomputation).

Draws the calibrated court, net and antennas on every frame of [--start, --start +
--seconds), using data/calibration/<id>_track.npz (scripts/track_camera.py), with frame,
pan angle and tracking confidence in the corner. With --ball, also the tracked ball from
data/cache/<id>/ball_* (scripts/detect_ball.py, track_ball.py): grey boxes = raw
detections, green circle = detected ball, orange = bridged gap, cyan = 1 s trail, white
curve = fitted 3D flight (grey = low confidence). --ground adds the drop line to the
floor and the ball's ground trail. With --people, the person detections from
data/cache/<id>/person_detections.npz (scripts/detect_people.py): green = score >= 0.4,
orange = weaker. With --rallies (implies --ball), a banner with the rally number, serving
team and how it ended, and a marker at each touch / serve / floor contact (yellow = us,
blue = them, red = floor; thin = outside any rally). With --tracks, the people tracks
(scripts/track_people.py): a colour and
number per track, a dot at the feet, and a top-down court map with everyone's position.
Encoded as H.264 so any player opens it.

Usage:
  python scripts/render_overlay.py --video titan-rise_set1 [--start 60 --seconds 60 --width 1920]
  python scripts/render_overlay.py --video titan-rise_set1 --ball --ground --start 60
  python scripts/render_overlay.py --video titan-rise_set1 --people --start 455
  python scripts/render_overlay.py --video titan-rise_set1 --tracks --start 455
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
    ap.add_argument("--people", action="store_true", help="draw the person detections")
    ap.add_argument("--rallies", action="store_true",
                    help="rally banner, ball trail with an X per touch, green rings under our "
                         "players (implies --ball, --rings)")
    ap.add_argument("--rings", action="store_true",
                    help="green ring on the floor under each of our players (Hudl style)")
    ap.add_argument("--detections", action="store_true", help="with --ball: raw ball boxes")
    ap.add_argument("--tracks", action="store_true",
                    help="draw people tracks with ids and a court map")
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
    if args.rallies:
        args.ball = True
    if args.ball:
        from setter_eyes.ball.overlay import BallOverlay

        ball = BallOverlay(args.ball_dir or config.repo_path(paths["cache_dir"]) / entry.key,
                           pipeline["ball"]["arcs"]["min_confidence"],
                           track.camera(track.ref_frame), show_detections=args.detections,
                           trail_px=pipeline["overlay"]["trail_px"])
    people = None
    if args.people:
        from setter_eyes.ball.tracker import Detections

        people = Detections.load(config.repo_path(paths["cache_dir"]) / entry.key /
                                 "person_detections.npz")
        people_groups = people.by_frame()
    tracks = None
    if args.rallies:
        args.rings = True
    if args.tracks or args.rings:
        from setter_eyes.track.overlay import TracksOverlay, draw_court_map, draw_rings
        from setter_eyes.track.people import PeopleTracks
        from setter_eyes.track.roles import load_roles

        tdir = config.repo_path(paths["cache_dir"]) / entry.key
        roles_path = tdir / "people_roles.json"
        tracks = TracksOverlay(PeopleTracks.load(tdir / "people_tracks.npz"),
                               load_roles(roles_path) if roles_path.exists() else None)
    rallies = None
    if args.rallies:
        from setter_eyes.rally.overlay import RallyOverlay

        rallies = RallyOverlay(config.repo_path(paths["cache_dir"]) / entry.key)
    kind = ("rallies" if rallies else "ball" if ball else "tracks" if args.tracks
            else "people" if people else "calibration")
    name = {"rallies": "rallies", "ball": "ball", "tracks": "tracks", "people": "people",
            "calibration": "demo"}[kind]
    out = args.out or config.repo_path(paths["output_dir"]) / kind / f"{entry.key}_{name}.mp4"
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
            img = draw_court(frame.image, cam.scaled(s), model,
                             None if ball or people or tracks else label,
                             keypoints=False)
            if args.rings and tracks is not None:
                in_rally = rallies is not None and rallies.in_rally(frame.index)
                draw_rings(img, tracks, frame.index, cam, s, pipeline["track"], in_rally,
                           radius_m=pipeline["overlay"]["ring_radius_m"])
            if ball is not None:
                extra = ball.draw(img, frame.index, cam, s, ground=args.ground)
                if rallies is not None:
                    if tracks is not None:
                        draw_court_map(img, tracks, frame.index, pipeline["track"],
                                       rallies.in_rally(frame.index), ball.ball_xyz)
                    banner = rallies.draw(img, frame.index, s)
                    _text(img, f"f{frame.index}   {banner}")
                    _text(img, extra, line=2)
                else:
                    _text(img, f"f{frame.index}   {extra}")
            if people is not None:
                # Detections are on every Nth frame: show the latest processed one.
                k = int(np.searchsorted(people.frames, frame.index, side="right")) - 1
                idx = people_groups.get(int(people.frames[k]), []) if k >= 0 else []
                for i in idx:
                    x1, y1, x2, y2 = (people.det_box[i] * s).round().astype(int)
                    colour = (0, 255, 0) if people.det_score[i] >= 0.4 else (0, 165, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), colour, max(1, round(2 * s * 2)))
                if ball is None:
                    _text(img, f"{label}   {len(idx)} people")
            if tracks is not None and args.tracks:
                n_tracked = tracks.draw(img, frame.index, s)
                if ball is None and people is None:
                    _text(img, f"{label}   {n_tracked} near-side players"
                               if tracks.roles is not None else f"{label}   {n_tracked} tracked")
            enc.stdin.write(img.tobytes())
            n += 1
    finally:
        enc.stdin.close()
        enc.wait()
    print(f"wrote {n} frames ({n / fps:.0f} s) -> {out}")
    return 0


def _text(image: np.ndarray, label: str, line: int = 1) -> None:
    t = max(1, round(image.shape[1] / 960))
    for colour, width in (((0, 0, 0), 4 * t), ((255, 255, 255), t)):
        cv2.putText(image, label, (10 * t, 30 * t * line), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t,
                    colour, width, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
