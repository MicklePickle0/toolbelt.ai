"""Draw the tracked ball on video frames: detections, ball marker, image trail, the fitted 3D
arc, and optionally a drop line to the floor with a ground trail (CLAUDE.md step 5)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath, Detections
from setter_eyes.calibration.camera import Camera

COLOURS = {"detection": (160, 160, 160), "measured": (0, 255, 0), "bridged": (0, 165, 255),
           "trail": (255, 255, 0), "arc": (255, 255, 255), "arc_weak": (110, 110, 110),
           "ground": (120, 255, 120)}


class BallOverlay:
    def __init__(self, cache_dir: Path, min_confidence: float, trail_s: float = 1.0):
        self.path = BallPath.load(cache_dir / "ball_path.npz")
        self.dets = Detections.load(cache_dir / "ball_detections.npz")
        d = json.loads((cache_dir / "ball_arcs.json").read_text(encoding="utf-8"))
        self.fps = float(d["fps"])
        self.g = np.array([0.0, 0.0, -float(d["gravity"])])
        self.arcs = [Arc(**a) for a in d["arcs"]]
        self.min_conf = min_confidence
        step = max(1, float(np.median(np.diff(self.path.frames))))
        self.trail = max(1, round(trail_s * self.fps / step))
        self.groups = self.dets.by_frame()

    def _index(self, frame: int) -> int:
        return int(np.searchsorted(self.path.frames, frame, side="right")) - 1

    def arc_at(self, k: int) -> Arc | None:
        for arc in self.arcs:
            if arc.first <= k <= arc.last:
                return arc
        return None

    def draw(self, image: np.ndarray, frame: int, cam: Camera, s: float,
             ground: bool = False) -> str:
        """Draw onto image (already scaled by s; cam scaled to match) -> a status label."""
        k = self._index(frame)
        if k < 0:
            return ""
        t = thickness = max(1, round(image.shape[1] / 960))
        pf = int(self.path.frames[k])
        for i in self.groups.get(pf, []):
            x1, y1, x2, y2 = (self.dets.det_box[i] * s).round().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), COLOURS["detection"], 1)
        lo = max(0, k - self.trail)
        trail = self.path.uv[lo:k + 1] * s
        trail = trail[np.isfinite(trail).all(axis=1)]
        if len(trail) > 1:
            cv2.polylines(image, [trail.round().astype(np.int32)], False, COLOURS["trail"],
                          thickness, cv2.LINE_AA)
        label = ""
        arc = self.arc_at(k)
        if arc is not None:
            strong = arc.confidence >= self.min_conf
            ts = np.linspace(arc.t0, arc.t0 + arc.duration_s, 60)
            pts = arc.position(ts, self.g)
            uv, z = cam.project(pts)
            uv = uv[z > 0.1]
            if len(uv) > 1:
                cv2.polylines(image, [uv.round().astype(np.int32)], False,
                              COLOURS["arc" if strong else "arc_weak"], t, cv2.LINE_AA)
            now = min(max(frame / self.fps, arc.t0), arc.t0 + arc.duration_s)
            p = arc.position(np.array([now]), self.g)[0]
            v = np.array(arc.v0) + self.g * (now - arc.t0)
            label = (f"ball z {p[2]:.1f} m  {np.linalg.norm(v):.0f} m/s  "
                     f"arc conf {arc.confidence:.2f}")
            if ground and strong:
                floor = np.array([p[0], p[1], 0.0])
                (bu, fu), zz = cam.project(np.stack([p, floor]))
                if (zz > 0.1).all():
                    cv2.line(image, tuple(bu.round().astype(int)), tuple(fu.round().astype(int)),
                             COLOURS["ground"], t, cv2.LINE_AA)
                past = arc.position(np.linspace(arc.t0, now, 30), self.g) * [1.0, 1.0, 0.0]
                guv, gz = cam.project(past)
                guv = guv[gz > 0.1]
                if len(guv) > 1:
                    cv2.polylines(image, [guv.round().astype(np.int32)], False,
                                  COLOURS["ground"], t, cv2.LINE_AA)
        if np.isfinite(self.path.uv[k]).all():
            u, v = (self.path.uv[k] * s).round().astype(int)
            r = max(6, round(float(np.nan_to_num(self.path.size_px[k], nan=30.0)) * s * 0.7))
            cv2.circle(image, (u, v), r, COLOURS["measured" if self.path.measured[k]
                                              else "bridged"], t + 1, cv2.LINE_AA)
        return label
