"""Check whether the camera moved (not just panned) after calibration.

Needs data/calibration/<id>.json and <id>_track.npz (scripts/track_camera.py). Writes
data/calibration/<id>_alignment.json and prints, per video, the background offset over
time and when the camera moved. A moved camera needs a new calibration from that point
on; until then, court positions and ball 3D after the move are off.
Settings: `calibration.alignment:` in config/court.yaml.

Usage:
  python scripts/check_camera.py --video mac-chrome_set1 [--every 30]
  python scripts/check_camera.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from setter_eyes import config
from setter_eyes.calibration import load
from setter_eyes.calibration.alignment import check_alignment
from setter_eyes.calibration.track import PoseTrack
from setter_eyes.labels import load_manifest


def _clock(frame: int, fps: float) -> str:
    m, s = divmod(frame / fps, 60)
    return f"{int(m)}:{int(s):02d}"


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="manifest key")
    ap.add_argument("--all", action="store_true", help="every video with a pan track")
    ap.add_argument("--every", type=float, help="seconds between samples (default: config)")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()
    if not args.video and not args.all:
        ap.error("--video or --all")

    cfg = config.load("court")["calibration"]
    cal_dir = config.repo_path(paths["calibration_dir"])
    manifest = load_manifest(args.manifest)
    keys = [args.video] if args.video else \
        [v.key for v in manifest.videos if (cal_dir / f"{v.key}_track.npz").exists()]
    for key in keys:
        entry = manifest.video(key)
        cal = load(key, cal_dir)
        track_path = cal_dir / f"{key}_track.npz"
        if cal is None or cal.frame is None or not track_path.exists():
            print(f"{key}: no calibration or pan track; skipped")
            continue
        report = check_alignment(entry.path, cal, PoseTrack.load(track_path), cfg, args.every)
        out = cal_dir / f"{key}_alignment.json"
        out.write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
        fps = report.fps
        trace = " ".join("-" if s.offset_px is None else f"{s.offset_px:.0f}"
                         for s in report.samples)
        print(f"{key}: {report.verdict.upper()}  offsets (px, every sample): {trace}")
        for lo, hi in report.moves:
            print(f"    camera moved between {_clock(lo, fps)} and {_clock(hi, fps)} "
                  f"(frames {lo}-{hi}): recalibrate from frame {hi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
