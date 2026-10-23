"""Pinhole camera with one radial distortion term, posed in court coordinates.

The court frame in config/court.yaml (x right, y toward the camera, z up) is
left-handed. OpenCV poses need a proper rotation, so the camera works on
mirrored points (x -> -x). That mirror lives here and in `to_cv`; callers
always pass and get plain court coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIRROR = np.array([-1.0, 1.0, 1.0])


def to_cv(points: np.ndarray) -> np.ndarray:
    """Court coordinates -> the right-handed frame OpenCV's pose refers to."""
    return np.asarray(points, float).reshape(-1, 3) * MIRROR


@dataclass
class Camera:
    width: int
    height: int
    f: float                 # px
    rvec: np.ndarray         # Rodrigues, mirrored court frame -> camera
    tvec: np.ndarray
    k1: float = 0.0          # radial distortion on normalised coordinates

    def __post_init__(self) -> None:
        self.rvec = np.asarray(self.rvec, float).reshape(3)
        self.tvec = np.asarray(self.tvec, float).reshape(3)

    @property
    def principal_point(self) -> np.ndarray:
        return np.array([self.width / 2.0, self.height / 2.0])

    @property
    def K(self) -> np.ndarray:
        cx, cy = self.principal_point
        return np.array([[self.f, 0, cx], [0, self.f, cy], [0, 0, 1.0]])

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def centre(self) -> np.ndarray:
        """Camera position in court coordinates (metres)."""
        return (-self.R.T @ self.tvec) * MIRROR

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / 2 / self.f)))

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Court points (N, 3) -> pixels (N, 2) and depth along the optical axis (N,).

        Points with depth <= 0 are behind the camera; their pixels are meaningless.
        """
        pc = to_cv(points) @ self.R.T + self.tvec
        z = pc[:, 2]
        xy = pc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
        if self.k1:
            xy = xy * (1 + self.k1 * (xy**2).sum(axis=1))[:, None]
        return xy * self.f + self.principal_point, z

    def scaled(self, s: float) -> Camera:
        """The same camera for an image resized by s (for drawing on previews)."""
        return Camera(round(self.width * s), round(self.height * s), self.f * s, self.rvec,
                      self.tvec, self.k1)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "f": float(self.f),
                "k1": float(self.k1), "rvec": self.rvec.tolist(), "tvec": self.tvec.tolist(),
                "centre_m": self.centre.round(3).tolist(), "hfov_deg": round(self.hfov_deg, 2)}

    @classmethod
    def from_dict(cls, d: dict) -> Camera:
        return cls(int(d["width"]), int(d["height"]), float(d["f"]), d["rvec"], d["tvec"],
                   float(d.get("k1", 0.0)))
