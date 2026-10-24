"""Calibration results on disk: data/calibration/<video>.json.

Kept outside data/cache because manual clicks are human work, not regenerable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from setter_eyes.calibration.camera import Camera


@dataclass
class Calibration:
    video: str
    camera: Camera
    method: str                                   # auto | manual
    confidence: float
    net_height_m: float
    frame: int | None = None                      # exact frame index the camera was fitted on
    support: dict = field(default_factory=dict)
    clicks: dict[str, list[float]] = field(default_factory=dict)   # keypoint -> [u, v]
    line_clicks: dict[str, list[list[float]]] = field(default_factory=dict)  # line -> [[u, v]]
    reprojection_rms_px: float | None = None
    needs_review: bool = False

    def to_dict(self) -> dict:
        return {"video": self.video, "method": self.method,
                "confidence": round(self.confidence, 4), "needs_review": self.needs_review,
                "net_height_m": self.net_height_m, "frame": self.frame,
                "camera": self.camera.to_dict(), "support": self.support, "clicks": self.clicks,
                "line_clicks": self.line_clicks,
                "reprojection_rms_px": self.reprojection_rms_px,
                "saved_at": datetime.now(UTC).isoformat(timespec="seconds")}

    @classmethod
    def from_dict(cls, d: dict) -> Calibration:
        return cls(video=d["video"], camera=Camera.from_dict(d["camera"]), method=d["method"],
                   confidence=d["confidence"], net_height_m=d["net_height_m"],
                   frame=d.get("frame"), support=d.get("support", {}),
                   clicks=d.get("clicks", {}), line_clicks=d.get("line_clicks", {}),
                   reprojection_rms_px=d.get("reprojection_rms_px"),
                   needs_review=d.get("needs_review", False))


def save(cal: Calibration, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{cal.video}.json"
    path.write_text(json.dumps(cal.to_dict(), indent=2), encoding="utf-8")
    return path


def load(video: str, directory: Path) -> Calibration | None:
    path = directory / f"{video}.json"
    if not path.exists():
        return None
    return Calibration.from_dict(json.loads(path.read_text(encoding="utf-8")))


def sample_frames(path: str | Path, n: int, start: float = 0.05,
                  end: float = 0.95) -> list[np.ndarray]:
    """n frames spread over the video by seeking. Not frame-accurate; only for a fixed
    camera's background, never for labelled frames or tracking."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open video {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        frames = []
        for frac in np.linspace(start, end, n):
            cap.set(cv2.CAP_PROP_POS_MSEC, frac * duration * 1000)
            ok, image = cap.read()
            if ok:
                frames.append(image)
        return frames
    finally:
        cap.release()


def median_background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median: moving players disappear, the court and lines stay.
    Only valid for a camera that doesn't move (the Humber footage pans)."""
    if not frames:
        raise ValueError("no frames")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
