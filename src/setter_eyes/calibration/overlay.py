"""Draw the calibrated court (floor lines, net tape at the game's height, antennas) on a frame."""

from __future__ import annotations

import cv2
import numpy as np

from setter_eyes.calibration.camera import Camera
from setter_eyes.calibration.court_model import FLOOR_LINES, CourtModel

COLOURS = {"floor": (0, 255, 255), "net": (255, 0, 255), "keypoint": (0, 128, 255)}


def _polyline(image, camera: Camera, a: np.ndarray, b: np.ndarray, colour, thickness) -> None:
    pts = np.linspace(a, b, 60)
    uv, z = camera.project(pts)
    run: list[np.ndarray] = []
    for p, d in zip(uv, z, strict=True):
        if d > 0.1 and np.all(np.abs(p) < 1e5):
            run.append(p)
        elif run:
            _draw_run(image, run, colour, thickness)
            run = []
    if run:
        _draw_run(image, run, colour, thickness)


def _draw_run(image, run, colour, thickness) -> None:
    if len(run) > 1:
        cv2.polylines(image, [np.round(np.array(run)).astype(np.int32)], False, colour,
                      thickness, cv2.LINE_AA)


def draw_court(image: np.ndarray, camera: Camera, model: CourtModel, label: str | None = None,
               keypoints: bool = True) -> np.ndarray:
    out = image.copy()
    t = max(1, round(image.shape[1] / 960))
    for name, (a, b) in model.lines.items():
        colour = COLOURS["floor"] if name in FLOOR_LINES else COLOURS["net"]
        _polyline(out, camera, a, b, colour, t)
    if keypoints:
        uv, z = camera.project(np.array(list(model.keypoints.values())))
        for p, d in zip(uv, z, strict=True):
            if d > 0.1 and 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
                cv2.circle(out, (round(p[0]), round(p[1])), 3 * t, COLOURS["keypoint"], -1,
                           cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (10 * t, 30 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t, (0, 0, 0),
                    4 * t, cv2.LINE_AA)
        cv2.putText(out, label, (10 * t, 30 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t,
                    (255, 255, 255), t, cv2.LINE_AA)
    return out
