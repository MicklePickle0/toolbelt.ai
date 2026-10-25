"""Follow the camera's pan through a whole video, starting from its calibration.

Needs data/calibration/<id>.json (scripts/calibrate.py). Decodes every frame once and writes
data/calibration/<id>_track.npz (a pose per frame + confidence), plus, for checking by eye:
  data/outputs/calibration/<id>_track.mp4    court overlay for --preview-start/--preview-seconds
  data/outputs/calibration/<id>_stills/      an overlaid frame every --stills-every seconds

If the camera was moved part-way (scripts/check_camera.py) and recalibrated after the
move (scripts/calibrate.py --segment), the later calibration takes over from there.
--reanchor applies new later calibrations to an existing track in seconds (no decoding),
and overlays the court on cached frames across the video into
data/outputs/calibration/<id>_reanchor/ for checking by eye.

Usage:
  python scripts/track_camera.py --video KEY [--preview-start 120 --preview-seconds 30]
  python scripts/track_camera.py --video KEY --reanchor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from setter_eyes import config
from setter_eyes.calibration import court_model, draw_court, load, load_segments
from setter_eyes.calibration.track import PoseTrack, reanchor, track_video
from setter_eyes.ingest import probe
from setter_eyes.labels import load_manifest


def _reanchor_only(entry, cal_dir: Path, later, paths: dict) -> int:
    from setter_eyes.labels.boxes import frame_from_name

    track_path = cal_dir / f"{entry.key}_track.npz"
    if not track_path.exists():
        raise SystemExit(f"no track at {track_path}; run without --reanchor first")
    track = reanchor(PoseTrack.load(track_path), later)
    track.save(track_path)
    print(f"{entry.key}: {len(later)} later calibration(s)"
          + "".join(f"\n  from frame {c.applies_from} (fitted on frame {c.frame})"
                    for c in later) + f"\n-> {track_path}")
    # Check by eye on exact cached frames spread over the video.
    pattern = config.load("labels")["frame_name_patterns"]
    files = sorted((config.repo_path(paths["cache_dir"]) / entry.key / "frames").glob("*.jpg"))
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(12, len(files))).astype(int)]
    out = config.repo_path(paths["output_dir"]) / "calibration" / f"{entry.key}_reanchor"
    out.mkdir(parents=True, exist_ok=True)
    model = court_model(entry.net_height_m)
    for f in files:
        idx = frame_from_name(f.name, pattern)
        img = cv2.imread(str(f))
        s = 1920 / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f.name), draw_court(img, track.camera(idx).scaled(s), model,
                                                  f"f{idx}", keypoints=False))
    print(f"overlays on cached frames -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths = config.load("pipeline")["paths"]
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--preview-start", type=float, default=60.0, help="seconds")
    ap.add_argument("--preview-seconds", type=float, default=30.0)
    ap.add_argument("--stills-every", type=float, default=60.0, help="seconds")
    ap.add_argument("--reanchor", action="store_true",
                    help="only apply later calibrations to the existing track")
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    cal_dir = config.repo_path(paths["calibration_dir"])
    cal = load(entry.key, cal_dir)
    if cal is None or cal.frame is None:
        raise SystemExit(f"no frame calibration for {entry.key}; run scripts/calibrate.py")
    later = load_segments(entry.key, cal_dir)
    if args.reanchor:
        return _reanchor_only(entry, cal_dir, later, paths)
    cfg = config.load("court")["calibration"]
    fps = probe(entry.path).fps
    p0 = round(args.preview_start * fps)
    p1 = p0 + round(args.preview_seconds * fps)
    still_every = max(1, round(args.stills_every * fps))
    # Frames arrive at the tracking width; previews and stills are drawn at that size.
    s_prev = cfg["track"]["width"] / cal.camera.width

    # Overlays need absolute poses, known only after the pass: keep JPEG-encoded copies.
    preview: dict[int, bytes] = {}
    stills: dict[int, bytes] = {}

    def on_frame(frame, _R, _conf):
        if p0 <= frame.index < p1:
            preview[frame.index] = cv2.imencode(".jpg", frame.image)[1].tobytes()
        if frame.index % still_every == 0:
            stills[frame.index] = cv2.imencode(".jpg", frame.image,
                                               [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()

    print(f"tracking {entry.key} from calibrated frame {cal.frame} ({entry.path.name})")
    track = track_video(entry.path, cal, cfg, on_frame=on_frame,
                        progress=lambda i: print(f"  frame {i}", flush=True))
    if later:
        track = reanchor(track, later)
        print(f"later calibrations from frames {[c.applies_from for c in later]}")
    saved = track.save(cal_dir / f"{entry.key}_track.npz")

    conf = track.confidence
    low = float((conf < cfg["track"]["min_confidence"]).mean())
    # Angle of each frame's rotation relative to the first (subtracting rotation vectors
    # breaks near 180 deg, where the vector flips sign).
    rot = Rotation.from_rotvec(track.rvecs)
    pan = np.degrees((rot * rot[0].inv()).magnitude())
    print(f"saved {saved}: {len(conf)} frames, mean confidence {conf.mean():.2f}, "
          f"{low:.1%} below {cfg['track']['min_confidence']}, pan range {pan.max():.1f} deg")

    model = court_model(entry.net_height_m)
    out = config.repo_path(paths["output_dir"]) / "calibration"
    out.mkdir(parents=True, exist_ok=True)

    def overlaid(idx: int, data: bytes, s: float) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        i = int(np.searchsorted(track.frames, idx))
        label = f"f{idx}  conf {conf[i]:.2f}"
        return draw_court(img, track.camera(idx).scaled(s), model, label, keypoints=False)

    if preview:
        first = next(iter(preview.values()))
        h, w = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR).shape[:2]
        mp4 = out / f"{entry.key}_track.mp4"
        writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for idx in sorted(preview):
            writer.write(overlaid(idx, preview[idx], s_prev))
        writer.release()
        print(f"preview {mp4}")
    still_dir = out / f"{entry.key}_stills"
    still_dir.mkdir(exist_ok=True)
    for idx, data in stills.items():
        cv2.imwrite(str(still_dir / f"{entry.key}_f{idx:06d}.jpg"), overlaid(idx, data, s_prev))
    print(f"stills {still_dir} ({len(stills)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
