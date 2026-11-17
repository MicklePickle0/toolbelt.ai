"""Camera pose from clicked court points and line points (the manual calibration).

Shared by the OpenCV click UI (scripts/calibrate.py) and the web app's calibration page:
both collect {keypoint: [u, v]} and {line: [[u, v], ...]} in full-resolution pixels and
solve them here, so a calibration made either way is the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from setter_eyes import config
from setter_eyes.calibration.camera import Camera
from setter_eyes.calibration.court_model import CourtModel
from setter_eyes.calibration.fit import fit_points_and_lines, refine_on_lines, solve_from_points
from setter_eyes.calibration.lines import LineScorer
from setter_eyes.calibration.store import Calibration


@dataclass
class ManualSolve:
    camera: Camera
    rms: float                  # px, clicked corner points
    line_rms: float             # px, clicked line points
    n_points: int
    n_line: int

    @property
    def worst_rms(self) -> float:
        return max(self.rms, self.line_rms)


def check_clicks(clicks: dict, line_clicks: dict, cfg: dict) -> str | None:
    """Why these clicks can't be solved, or None if they can."""
    n_line = sum(len(v) for v in line_clicks.values())
    if len(clicks) < 4 or len(clicks) + n_line < cfg["manual_min_points"]:
        return (f"need at least 4 corner points and {cfg['manual_min_points']} clicks in "
                f"total, got {len(clicks)} + {n_line}")
    return None


def solve_manual(width: int, height: int, model: CourtModel, cfg: dict,
                 clicks: dict[str, list[float]], line_clicks: dict[str, list[list[float]]],
                 scorer: LineScorer | None = None) -> ManualSolve:
    """Clicks -> camera. Raises ValueError if there are too few (check_clicks)."""
    problem = check_clicks(clicks, line_clicks, cfg)
    if problem:
        raise ValueError(problem)
    unknown = [n for n in clicks if n not in model.keypoints] + \
        [n for n in line_clicks if n not in model.lines]
    if unknown:
        raise ValueError(f"unknown court points or lines: {unknown}")
    hfov = tuple(cfg["hfov_range_deg"])
    n_line = sum(len(v) for v in line_clicks.values())
    world = np.array([model.keypoints[n] for n in clicks])
    pixels = np.array(list(clicks.values()), float)
    cam, _ = solve_from_points(world, pixels, width, height, hfov)
    segments = [model.lines[n] for n in line_clicks]
    pts = [np.array(v, float) for v in line_clicks.values()]
    k1 = cfg["estimate_distortion"] and len(clicks) + n_line >= 12
    cam, rms, line_rms = fit_points_and_lines(cam, world, pixels, segments, pts, hfov, k1)
    if cfg.get("manual_refine_on_mask") and scorer is not None:
        cam = refine_on_lines(cam, scorer, cfg, estimate_k1=k1, points=(world, pixels))
        rms = float(np.sqrt(((cam.project(world)[0] - pixels) ** 2).sum(axis=1).mean()))
    return ManualSolve(cam, float(rms), float(line_rms), len(clicks), n_line)


def error_limit_px(cfg: dict, width: int) -> float:
    """Above this reprojection error (px at this width) a manual solve is flagged."""
    return cfg["max_reprojection_error_px"] * config.pixel_scale(width)


def manual_calibration(key: str, frame: int, width: int, model: CourtModel, cfg: dict,
                       solved: ManualSolve, clicks: dict, line_clicks: dict,
                       support: dict | None = None) -> Calibration:
    ok = solved.worst_rms <= error_limit_px(cfg, width)
    return Calibration(video=key, camera=solved.camera, method="manual",
                       confidence=1.0 if ok else 0.5, net_height_m=model.net_height,
                       frame=frame, support=support or {}, clicks=clicks,
                       line_clicks=line_clicks,
                       reprojection_rms_px=round(solved.worst_rms, 2), needs_review=not ok)
