"""Camera fitting: from clicked keypoints (solvePnP + focal search) and refinement on lines."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from setter_eyes.calibration.camera import Camera, to_cv
from setter_eyes.calibration.lines import LineScorer

K1_BOUNDS = (-0.5, 0.5)


def focal_from_hfov(width: int, hfov_deg: float) -> float:
    return width / 2 / np.tan(np.radians(hfov_deg) / 2)


def _pack(cam: Camera, with_k1: bool) -> np.ndarray:
    x = [np.log(cam.f), *cam.rvec, *cam.tvec]
    return np.array([*x, cam.k1] if with_k1 else x)


def _unpack(x: np.ndarray, like: Camera, with_k1: bool) -> Camera:
    return Camera(like.width, like.height, float(np.exp(x[0])), x[1:4], x[4:7],
                  float(x[7]) if with_k1 else like.k1)


def _bounds(cam: Camera, hfov_range: tuple[float, float], with_k1: bool):
    lo_f, hi_f = (np.log(focal_from_hfov(cam.width, h)) for h in sorted(hfov_range, reverse=True))
    lo = [lo_f] + [-np.inf] * 6 + ([K1_BOUNDS[0]] if with_k1 else [])
    hi = [hi_f] + [np.inf] * 6 + ([K1_BOUNDS[1]] if with_k1 else [])
    return np.array(lo), np.array(hi)


def solve_from_points(world: np.ndarray, image: np.ndarray, width: int, height: int,
                      hfov_range: tuple[float, float] = (30, 120),
                      steps: int = 46) -> tuple[Camera, float]:
    """Pose + focal length from >= 4 court/pixel correspondences -> (camera, RMS px)."""
    world = np.asarray(world, float).reshape(-1, 3)
    image = np.asarray(image, float).reshape(-1, 2)
    if len(world) < 4:
        raise ValueError("need at least 4 points")
    obj = to_cv(world)
    best: tuple[float, Camera] | None = None
    for hfov in np.linspace(*hfov_range, steps):
        cam = Camera(width, height, focal_from_hfov(width, hfov), np.zeros(3), np.zeros(3))
        ok, rvec, tvec = cv2.solvePnP(obj, image, cam.K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj, image, cam.K, None, rvec, tvec)
        cam = Camera(width, height, cam.f, rvec, tvec)
        rms = _rms(cam, world, image)
        if np.isfinite(rms) and (best is None or rms < best[0]):
            best = (rms, cam)
    if best is None:
        raise ValueError("solvePnP failed for every focal length")

    cam = best[1]
    sol = least_squares(lambda x: (_unpack(x, cam, False).project(world)[0] - image).ravel(),
                        _pack(cam, False), bounds=_bounds(cam, hfov_range, False))
    cam = _unpack(sol.x, cam, False)
    return cam, _rms(cam, world, image)


def _rms(cam: Camera, world: np.ndarray, image: np.ndarray) -> float:
    uv, z = cam.project(world)
    if (z <= 0).any():
        return float("inf")
    return float(np.sqrt(((uv - image) ** 2).sum(axis=1).mean()))


def refine_on_lines(cam: Camera, scorer: LineScorer, cfg: dict, estimate_k1: bool = False,
                    points: tuple[np.ndarray, np.ndarray] | None = None,
                    point_sigma_px: float = 3.0) -> Camera:
    """Move the camera so projected court lines sit on the line mask (coarse to fine).

    `points` (court, pixel) anchors a manual calibration so refinement can't wander off.
    """
    uv, z = cam.project(scorer.pts)
    _, vis = scorer.lookup(uv, z)
    pts = scorer.pts[vis]
    if len(pts) < 20:
        return cam
    cap = float(cfg["refine_cap_px"])
    tol = scorer.tol
    hfov_range = tuple(cfg["hfov_range_deg"])

    for cap_px, sigma in ((cap, 3.0), (cap / 3, 1.0)):
        field = cv2.GaussianBlur(np.minimum(scorer.dist, cap_px), (0, 0), sigma)

        def residuals(x, field=field, cap_px=cap_px, like=cam):
            c = _unpack(x, like, estimate_k1)
            uv, z = c.project(pts)
            uvs = uv * scorer.scale
            vals = map_coordinates(field, [uvs[:, 1], uvs[:, 0]], order=1, mode="nearest")
            outside = (z <= 0.1) | (uvs[:, 0] < 0) | (uvs[:, 1] < 0) | \
                (uvs[:, 0] > scorer.w - 1) | (uvs[:, 1] > scorer.h - 1)
            r = np.where(outside, cap_px, vals) / tol
            if points is not None:
                world, image = points
                rp = (c.project(world)[0] - image).ravel() / point_sigma_px
                r = np.concatenate([r, rp])
            return r

        sol = least_squares(residuals, _pack(cam, estimate_k1), loss="soft_l1", f_scale=1.0,
                            x_scale="jac", max_nfev=200,
                            bounds=_bounds(cam, hfov_range, estimate_k1))
        cam = _unpack(sol.x, cam, estimate_k1)
    return cam
