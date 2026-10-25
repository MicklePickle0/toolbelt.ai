"""Carry a calibration over to a later frame of the same video, without clicks.

For when the camera was moved or knocked (scripts/check_camera.py): the court hasn't
changed, so floor features in the calibrated frame (line junctions, logos, wood marks)
have known court positions: their rays from the calibrated camera hit the floor (z = 0).
Matched into the later frame (SIFT, ratio test), they give 2D-3D pairs, and
cv2.solvePnPRansac gives the new pose, translation included. Focal length and lens
distortion are kept (a zoom change fails the fit and needs clicking).

Players, reflections on the glossy floor and off-floor features are outliers to RANSAC.
Confidence from the inlier count, the reprojection error and how widely the inliers
spread over the floor (a pose fitted on one corner is poorly constrained).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from setter_eyes.calibration.camera import Camera, to_cv


@dataclass
class TransferResult:
    camera: Camera
    inliers: int
    rms_px: float
    spread_m: tuple[float, float]       # extent of the inliers on the floor (x, y)
    confidence: float

    def to_dict(self) -> dict:
        return {"inliers": self.inliers, "rms_px": round(self.rms_px, 2),
                "spread_m": [round(v, 1) for v in self.spread_m],
                "confidence": round(self.confidence, 3)}


def transfer_calibration(cam: Camera, image_ref: np.ndarray, image_new: np.ndarray,
                         cfg: dict) -> TransferResult | None:
    """cam fits image_ref; -> the camera for image_new (same size), or None."""
    t = cfg["transfer"]
    sift = cv2.SIFT_create(t["features"])
    ka, da = sift.detectAndCompute(cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return None
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < t["ratio"] * p[1].distance]
    if len(good) < t["min_inliers"]:
        return None
    ua = np.array([ka[m.queryIdx].pt for m in good])
    ub = np.array([kb[m.trainIdx].pt for m in good])

    # Floor positions from the calibrated camera.
    rays = cam.rays(ua)
    c = cam.centre
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -c[2] / rays[:, 2]
    floor = c + lam[:, None] * rays
    (x0, x1), (y0, y1) = t["floor_x_m"], t["floor_y_m"]
    on = (rays[:, 2] < -t["min_ray_down"]) & (lam > 0) & (floor[:, 0] > x0) & \
        (floor[:, 0] < x1) & (floor[:, 1] > y0) & (floor[:, 1] < y1)
    if on.sum() < t["min_inliers"]:
        return None
    P, uv = floor[on], ub[on]
    dist = np.array([cam.k1, 0.0, 0.0, 0.0])
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        to_cv(P), uv, cam.K, dist, reprojectionError=t["ransac_px"],
        iterationsCount=t["iterations"], flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < t["min_inliers"]:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(to_cv(P[inl]), uv[inl], cam.K, dist, rvec, tvec)
    new = Camera(cam.width, cam.height, cam.f, rvec.ravel(), tvec.ravel(), cam.k1)
    proj, _ = new.project(P[inl])
    rms = float(np.sqrt(np.mean(np.sum((proj - uv[inl]) ** 2, axis=1))))
    spread = (float(np.ptp(P[inl, 0])), float(np.ptp(P[inl, 1])))
    conf = min(1.0, len(inl) / t["full_inliers"]) * float(np.exp(-rms / t["rms_scale_px"])) * \
        min(1.0, spread[0] / t["full_spread_m"]) * min(1.0, spread[1] / t["full_spread_m"])
    return TransferResult(new, len(inl), rms, spread, conf)
