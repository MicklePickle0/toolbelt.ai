"""Fix the pan track where scripts/check_camera.py found the camera moved: automatically.

For each move in data/calibration/<id>_alignment.json, the calibration is carried over
(setter_eyes.calibration.transfer) to the first cached frame after the move (trying a few
if needed), saved as an extra calibration (data/calibration/<id>@f<frame>.json, method
"transfer"), and the pan track is re-anchored. Then the video is checked again; a move
still found (e.g. a second one) gets the same treatment, up to --rounds times.

A transfer below calibration.transfer.min_confidence isn't used: click that one instead
(scripts/calibrate.py --video KEY --frame N --segment --manual). Overlays of each new
calibration: data/outputs/calibration/<id>@f<frame>.jpg.
Afterwards, rerun what depends on court positions: track_ball.py, track_people.py,
find_rallies.py (not detection).

Usage:
  python scripts/fix_camera.py --video mac-chrome_set1
  python scripts/fix_camera.py --all            # every video whose last check says moved
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from setter_eyes import config
from setter_eyes.calibration import (
    Calibration,
    court_model,
    draw_court,
    load,
    load_segments,
    save,
)
from setter_eyes.calibration.alignment import check_alignment
from setter_eyes.calibration.track import PoseTrack, reanchor
from setter_eyes.calibration.transfer import transfer_calibration
from setter_eyes.labels import load_manifest
from setter_eyes.labels.boxes import frame_from_name


def _clock(frame: int, fps: float) -> str:
    m, s = divmod(frame / fps, 60)
    return f"{int(m)}:{int(s):02d}"


def _source(cal: Calibration, later: list[Calibration], frame: int) -> Calibration:
    """The calibration in force at `frame` (the one to carry over)."""
    src = cal
    for c in later:
        if c.applies_from <= frame:
            src = c
    return src


def fix_video(entry, cfg: dict, paths: dict, rounds: int) -> None:
    cal_dir = config.repo_path(paths["calibration_dir"])
    cal = load(entry.key, cal_dir)
    track_path = cal_dir / f"{entry.key}_track.npz"
    report_path = cal_dir / f"{entry.key}_alignment.json"
    if cal is None or cal.frame is None or not track_path.exists() or not report_path.exists():
        print(f"{entry.key}: needs a calibration, pan track and check_camera.py report")
        return
    pattern = config.load("labels")["frame_name_patterns"]
    frames_dir = config.repo_path(paths["cache_dir"]) / entry.key / "frames"
    cached = sorted((frame_from_name(f.name, pattern), f) for f in frames_dir.glob("*.jpg"))
    t = cfg["transfer"]
    model = court_model(entry.net_height_m)
    out_dir = config.repo_path(paths["output_dir"]) / "calibration"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fps = report["fps"]
    for _ in range(rounds):
        moves = report["moves"]
        if not moves:
            print(f"{entry.key}: {report['verdict'].upper()}, no moves left")
            return
        later = load_segments(entry.key, cal_dir)
        done_starts = {c.applies_from for c in later}
        added = 0
        for _lo, hi in moves:
            if hi in done_starts:
                continue
            src = _source(cal, later, hi)
            ref_img = cv2.imread(str(next(f for i, f in cached if i == src.frame)))
            after = [(i, f) for i, f in cached if i >= hi + round(t["settle_s"] * fps)]
            best = None
            for i, f in after[:t["max_tries"]]:
                img = cv2.imread(str(f))
                res = transfer_calibration(src.camera, ref_img, img, cfg)
                if res is not None and (best is None or res.confidence > best[1].confidence):
                    best = (i, res, img)
                if best is not None and best[1].confidence >= t["good_confidence"]:
                    break
            if not after:
                print(f"  move at {_clock(hi, fps)}: no cached frame after it; extract some "
                      f"(extract_frames.py --video {entry.key} --start {hi / fps + 3:.0f} "
                      f"--end {hi / fps + 4:.0f} --every 30) and run again")
                continue
            if best is None or best[1].confidence < t["min_confidence"]:
                frame_hint = after[0][0]
                print(f"  move at {_clock(hi, fps)}: transfer failed"
                      + (f" (best confidence {best[1].confidence:.2f})" if best else "")
                      + f"; click it: calibrate.py --video {entry.key} --frame {frame_hint} "
                      "--segment --manual")
                continue
            i, res, img = best
            new = Calibration(entry.key, res.camera, "transfer", res.confidence,
                              entry.net_height_m, frame=i, support=res.to_dict(),
                              applies_from=int(hi), needs_review=False)
            save(new, cal_dir, segment=True)
            name = f"{entry.key}@f{i:06d}"
            s = min(1.0, 1920 / img.shape[1])
            cv2.imwrite(str(out_dir / f"{name}.jpg"), cv2.resize(
                draw_court(img, res.camera, model, f"{name} transfer {res.confidence:.2f}"),
                None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            print(f"  move at {_clock(hi, fps)}: carried over to frame {i} "
                  f"({res.inliers} floor matches, rms {res.rms_px:.1f} px, "
                  f"confidence {res.confidence:.2f})")
            added += 1
        if not added:
            return
        track = reanchor(PoseTrack.load(track_path), load_segments(entry.key, cal_dir))
        track.save(track_path)
        rep = check_alignment(entry.path, cal, track, cfg)
        report = rep.to_dict()
        report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        trace = " ".join("-" if x.offset_px is None else f"{x.offset_px:.0f}"
                         for x in rep.samples)
        print(f"{entry.key}: after fixing: {rep.verdict.upper()}  offsets: {trace}")


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="manifest key")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rounds", type=int, default=3, help="fix, re-check, repeat")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()
    if not args.video and not args.all:
        ap.error("--video or --all")
    cfg = config.load("court")["calibration"]
    cal_dir = config.repo_path(paths["calibration_dir"])
    manifest = load_manifest(args.manifest)
    entries = [manifest.video(args.video)] if args.video else [
        v for v in manifest.videos if (cal_dir / f"{v.key}_alignment.json").exists() and
        json.loads((cal_dir / f"{v.key}_alignment.json").read_text())["moves"]]
    for entry in entries:
        print(f"== {entry.key}")
        fix_video(entry, cfg, paths, args.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
