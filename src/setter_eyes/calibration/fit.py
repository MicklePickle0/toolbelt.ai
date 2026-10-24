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


def line_click_distances(cam: Camera, segments: list[np.ndarray],
                         clicks: list[np.ndarray]) -> np.ndarray:
    """Pixel distance from each clicked point to the projected court line it was clicked on.
    segments[i] is that line's (2, 3) court endpoints, clicks[i] its (n, 2) pixels."""
    out = []
    for (a, b), pts in zip(segments, clicks, strict=True):
        uv, z = cam.project(np.linspace(a, b, 120))
        uv = uv[z > 0.1]
        if len(uv) < 2:
            out.append(np.full(len(pts), 1e3))
            continue
        p0, d = uv[:-1], uv[1:] - uv[:-1]
        length2 = (d ** 2).sum(axis=1) + 1e-12
        for q in np.asarray(pts, float):
            t = np.clip(((q - p0) * d).sum(axis=1) / length2, 0, 1)
            out.append([np.linalg.norm(p0 + t[:, None] * d - q, axis=1).min()])
    return np.concatenate([np.ravel(o) for o in out]) if out else np.zeros(0)


def fit_points_and_lines(cam: Camera, world: np.ndarray, pixels: np.ndarray,
                         segments: list[np.ndarray], clicks: list[np.ndarray],
                         hfov_range: tuple[float, float], estimate_k1: bool = False,
                         ) -> tuple[Camera, float, float]:
    """Refine pose + focal (+ k1) so keypoints land on their clicks AND each court line
    passes through the points clicked along it (for lines whose corners are hidden).
    -> (camera, keypoint RMS px, line-click RMS px)."""
    world = np.asarray(world, float).reshape(-1, 3)
    pixels = np.asarray(pixels, float).reshape(-1, 2)

    def residuals(x):
        c = _unpack(x, cam, estimate_k1)
        r = [(c.project(world)[0] - pixels).ravel()]
        if segments:
            r.append(line_click_distances(c, segments, clicks))
        return np.concatenate(r)

    x0 = _pack(cam, estimate_k1)
    lo, hi = _bounds(cam, hfov_range, estimate_k1)
    sol = least_squares(residuals, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                        loss="soft_l1", f_scale=3.0, x_scale="jac")
    cam = _unpack(sol.x, cam, estimate_k1)
    line_d = line_click_distances(cam, segments, clicks) if segments else np.zeros(0)
    line_rms = float(np.sqrt((line_d ** 2).mean())) if len(line_d) else 0.0
    return cam, _rms(cam, world, pixels), line_rms


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
