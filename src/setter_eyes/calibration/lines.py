"""Painted-line evidence: a line mask, a distance map, and a score for how well a
camera's projected court lines land on it.

Pixel settings are at `process_width` (frames are downscaled first); cameras are
always full resolution and their projections are scaled to the mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from setter_eyes.calibration.camera import Camera
from setter_eyes.calibration.court_model import CourtModel


def downscale(image: np.ndarray, process_width: int) -> tuple[np.ndarray, float]:
    """Resize so the width is at most process_width; returns (image, scale)."""
    s = min(1.0, process_width / image.shape[1])
    if s < 1.0:
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return image, s


def line_mask(image_bgr: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    """Thin, bright, unsaturated structures (painted lines, net tape) -> (uint8 mask, scale)."""
    small, s = downscale(image_bgr, cfg["process_width"])
    m = cfg["line_mask"]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    k = m["tophat_kernel_px"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(val, cv2.MORPH_TOPHAT, kernel)
    mask = (tophat >= m["min_tophat"]) & (sat <= m["max_saturation"]) & (val >= m["min_value"])
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                            np.ones((2, 2), np.uint8))
    return mask, s


@dataclass(frozen=True)
class Support:
    inliers: int
    visible: int
    support: float            # inliers / visible
    chance: float             # share of the image within tolerance of a line pixel
    lines_found: list[str]
    confidence: float         # support above chance, scaled by how many lines were found

    def to_dict(self) -> dict:
        return {"inliers": self.inliers, "visible": self.visible,
                "support": round(self.support, 4), "chance": round(self.chance, 4),
                "lines_found": self.lines_found, "confidence": round(self.confidence, 4)}


class LineScorer:
    def __init__(self, mask: np.ndarray, scale: float, model: CourtModel, cfg: dict,
                 step: float | None = None):
        self.mask = mask
        self.scale = scale
        self.model = model
        self.cfg = cfg
        self.tol = float(cfg["inlier_tolerance_px"])
        self.dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)
        self.chance = float((self.dist <= self.tol).mean())
        self.pts, self.line_idx = model.samples(step or cfg["sample_step_m"])
        self.h, self.w = mask.shape

    def lookup(self, uv_full: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the nearest line pixel at full-res pixels -> (dist, visible)."""
        uv = uv_full * self.scale
        u, v = uv[..., 0], uv[..., 1]
        visible = (depth > 0.1) & (u >= 0) & (v >= 0) & (u <= self.w - 1) & (v <= self.h - 1)
        ui = np.clip(np.nan_to_num(u), 0, self.w - 1).astype(np.int32)
        vi = np.clip(np.nan_to_num(v), 0, self.h - 1).astype(np.int32)
        return self.dist[vi, ui], visible

    def evaluate(self, camera: Camera) -> Support:
        uv, z = camera.project(self.pts)
        d, vis = self.lookup(uv, z)
        inl = vis & (d <= self.tol)
        n_lines = len(self.model.lines)
        vis_per = np.bincount(self.line_idx[vis], minlength=n_lines)
        inl_per = np.bincount(self.line_idx[inl], minlength=n_lines)
        names = self.model.line_names
        # A found line must also be long in the image: a court collapsed onto one
        # bright stripe would otherwise "find" every line.
        uv_small = uv * self.scale
        found = []
        for i in range(n_lines):
            if vis_per[i] < 8 or inl_per[i] < self.cfg["line_found_fraction"] * vis_per[i]:
                continue
            p = uv_small[inl & (self.line_idx == i)]
            if np.hypot(*np.ptp(p, axis=0)) >= self.cfg["min_found_line_px"]:
                found.append(names[i])
        visible = int(vis.sum())
        support = float(inl.sum() / visible) if visible else 0.0
        above_chance = max(0.0, (support - self.chance) / max(1e-6, 1 - self.chance))
        confidence = above_chance * min(1.0, len(found) / self.cfg["min_found_lines"])
        return Support(int(inl.sum()), visible, support, self.chance, found, confidence)
