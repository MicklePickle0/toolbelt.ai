"""Calibrate the camera of manifest videos on one exact frame: automatic line fitting first,
manual-click fallback.

The frame comes from data/cache/<id>/frames (run extract_frames.py first), so its index
is exact; scripts/track_camera.py follows the camera's pan from there. Below
calibration.auto_min_confidence (config/court.yaml) the click UI opens, unless --no-ui,
in which case the result is saved with needs_review = true.

Writes data/calibration/<id>.json and, for checking by eye,
data/outputs/calibration/<id>.jpg (overlay) and <id>_mask.png (line evidence).

Usage:
  python scripts/calibrate.py --video KEY [--frame N] [--manual] [--no-ui]
  python scripts/calibrate.py --all --no-ui       # try auto everywhere, click later
  python scripts/calibrate.py --missing --manual  # click every uncalibrated video in turn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from setter_eyes import config
from setter_eyes.calibration import (
    Calibration,
    LineScorer,
    auto_calibrate,
    court_model,
    draw_court,
    line_mask,
    refine_on_lines,
    save,
    solve_from_points,
)
from setter_eyes.calibration.court_model import CourtModel
from setter_eyes.calibration.fit import fit_points_and_lines
from setter_eyes.labels import load_manifest
from setter_eyes.labels.boxes import frame_from_name


def manual(image: np.ndarray, model: CourtModel, cfg: dict, key: str, frame: int,
           scorer: LineScorer) -> Calibration | None:
    from setter_eyes.calibration.manual import click_keypoints, click_lines, confirm

    hfov = tuple(cfg["hfov_range_deg"])
    while True:
        clicks = click_keypoints(image, model)
        if clicks is None:
            return None
        # Then points anywhere along lines whose corners are hidden (e.g. the near end line).
        line_clicks = click_lines(image, clicks)
        if line_clicks is None:
            return None
        n_line = sum(len(v) for v in line_clicks.values())
        if len(clicks) < 4 or len(clicks) + n_line < cfg["manual_min_points"]:
            print(f"  need at least 4 corner points and {cfg['manual_min_points']} clicks in "
                  f"total, got {len(clicks)} + {n_line}")
            continue
        world = np.array([model.keypoints[n] for n in clicks])
        pixels = np.array(list(clicks.values()))
        h, w = image.shape[:2]
        cam, _ = solve_from_points(world, pixels, w, h, hfov)
        segments = [model.lines[n] for n in line_clicks]
        pts = [np.array(v) for v in line_clicks.values()]
        k1 = cfg["estimate_distortion"] and len(clicks) + n_line >= 12
        cam, rms, line_rms = fit_points_and_lines(cam, world, pixels, segments, pts, hfov, k1)
        if cfg.get("manual_refine_on_mask"):
            cam = refine_on_lines(cam, scorer, cfg, estimate_k1=k1, points=(world, pixels))
            rms = float(np.sqrt(((cam.project(world)[0] - pixels) ** 2).sum(axis=1).mean()))
        sup = scorer.evaluate(cam)
        label = f"{key}  manual  points rms {rms:.1f}px  lines rms {line_rms:.1f}px"
        print(f"  {len(clicks)} points (rms {rms:.1f} px), {n_line} line clicks on "
              f"{len(line_clicks)} lines (rms {line_rms:.1f} px)")
        if confirm(draw_court(image, cam, model, label)):
            ok = max(rms, line_rms) <= cfg["max_reprojection_error_px"]
            return Calibration(video=key, camera=cam, method="manual",
                               confidence=1.0 if ok else 0.5, net_height_m=model.net_height,
                               frame=frame, support=sup.to_dict(), clicks=clicks,
                               line_clicks=line_clicks,
                               reprojection_rms_px=round(max(rms, line_rms), 2),
                               needs_review=not ok)


def reference_frame(cache_dir: Path, key: str, frame: int | None) -> tuple[int, Path]:
    files = sorted((cache_dir / key / "frames").glob(f"{key}_f*.jpg"))
    pattern = config.load("labels")["frame_name_patterns"]
    indexed = {frame_from_name(f.name, pattern): f for f in files}
    if not indexed:
        raise SystemExit(f"no cached frames for {key}; run scripts/extract_frames.py first")
    if frame is None:
        frame = min(indexed)
    if frame not in indexed:
        raise SystemExit(f"frame {frame} of {key} is not cached; cached: "
                         f"{sorted(indexed)[:10]}...")
    return frame, indexed[frame]


def calibrate_one(key: str, net_height: float, args, cfg: dict, paths: dict) -> Calibration | None:
    frame, image_path = reference_frame(config.repo_path(paths["cache_dir"]), key, args.frame)
    print(f"\n== {key}  frame {frame}  (net {net_height} m)")
    image = cv2.imread(str(image_path))
    model = court_model(net_height)
    mask, s = line_mask(image, cfg)
    out_dir = config.repo_path(paths["output_dir"]) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{key}_mask.png"), mask)
    scorer = LineScorer(mask, s, model, cfg)

    cal = None
    if not args.manual:
        res = auto_calibrate(image, model, cfg, mask, s)
        if res is None:
            print("  auto: no plausible court found")
        else:
            c = res.support.confidence
            print(f"  auto: confidence {c:.2f}, lines {res.support.lines_found}, "
                  f"camera at {res.camera.centre.round(2)} m, hfov {res.camera.hfov_deg:.0f}")
            cal = Calibration(video=key, camera=res.camera, method="auto", confidence=c,
                              net_height_m=net_height, frame=frame,
                              support=res.support.to_dict(),
                              needs_review=c < cfg["auto_min_confidence"])

    if (cal is None or cal.needs_review) and not args.no_ui:
        print("  opening the click UI")
        cal = manual(image, model, cfg, key, frame, scorer) or cal
    if cal is None:
        print("  NOT CALIBRATED (run again without --no-ui to click points)")
        return None

    label = f"{key}  f{frame}  {cal.method}  confidence {cal.confidence:.2f}"
    overlay = draw_court(image, cal.camera, model, label)
    s_out = min(1.0, 1920 / overlay.shape[1])
    cv2.imwrite(str(out_dir / f"{key}.jpg"),
                cv2.resize(overlay, None, fx=s_out, fy=s_out, interpolation=cv2.INTER_AREA))
    saved = save(cal, config.repo_path(paths["calibration_dir"]))
    flag = "  NEEDS REVIEW" if cal.needs_review else ""
    print(f"  saved {saved}{flag}\n  overlay {out_dir / f'{key}.jpg'}")
    return cal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths = config.load("pipeline")["paths"]
    ap.add_argument("--video", action="append", help="manifest key (repeatable)")
    ap.add_argument("--all", action="store_true", help="every video in the manifest")
    ap.add_argument("--missing", action="store_true",
                    help="every manifest video that has no calibration yet")
    ap.add_argument("--frame", type=int, help="cached frame index (default: the first cached)")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--manual", action="store_true", help="skip auto, click points")
    ap.add_argument("--no-ui", action="store_true", help="never open the click UI")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    entries = manifest.videos if args.all else [manifest.video(k) for k in args.video or []]
    if args.missing:
        cal_dir = config.repo_path(paths["calibration_dir"])
        entries = [v for v in manifest.videos if not (cal_dir / f"{v.key}.json").exists()]
        print(f"{len(entries)} videos without a calibration: "
              + ", ".join(v.key for v in entries))
    if not entries:
        ap.error("give --video KEY, --all or --missing (or everything is calibrated)")
    cfg = config.load("court")["calibration"]
    done = [calibrate_one(v.key, v.net_height_m, args, cfg, paths) for v in entries]
    ok = [c for c in done if c is not None]
    review = [c.video for c in ok if c.needs_review]
    print(f"\n{len(ok)}/{len(entries)} calibrated" +
          (f"; needs review: {', '.join(review)}" if review else ""))
    return 0 if len(ok) == len(entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
