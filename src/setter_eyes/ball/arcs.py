"""Split the ball path into flights and fit each one as a 3D ballistic arc.

Segmentation: grow a flight while a quadratic in time still fits the pan-free pixels
(a projected ballistic arc is close to a parabola over a flight); where it stops
fitting the ball was touched or bounced. Gaps longer than `max_gap_s` also split.

3D fit: p(t) = p0 + v0 t + 0.5 g t^2 in court metres (z up), projected through each
frame's camera, least squares on pixels. Gravity's known size is what fixes depth
(the metric scale) from one camera. Starting points come from a grid of depths along
the first and last rays. Ball apparent size is a cross-check only.

Confidence: pixel fit, flight duration (short flights fix depth poorly) and the
spread of the fitted depth across the grid of starting points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from setter_eyes.calibration.camera import Camera

BALL_DIAMETER_M = 0.21


@dataclass
class Arc:
    first: int                # index into the path
    last: int
    t0: float                 # seconds at path index `first`
    p0: list[float]           # court metres at t0
    v0: list[float]           # m/s
    rms_px: float
    duration_s: float
    n_obs: int
    size_ratio: float         # measured / expected ball size (cross-check; ~1 is good)
    confidence: float

    def position(self, t: np.ndarray, g: np.ndarray) -> np.ndarray:
        dt = np.asarray(t, float)[:, None] - self.t0
        return np.array(self.p0) + np.array(self.v0) * dt + 0.5 * g * dt**2

    def to_dict(self) -> dict:
        return asdict(self)


def segment(t: np.ndarray, uv: np.ndarray, cfg: dict) -> list[tuple[int, int]]:
    """Indices [i, j] (inclusive) of flights among valid samples (t, uv).

    A touch or bounce is a sudden change of velocity. At each sample, one quadratic
    through the `window_samples` either side is compared with two separate ones (before,
    after); where splitting there cuts the squared error by more than `break_px` per
    sample (RMS-equivalent) the ball changed course. Local windows keep perspective
    (which bends long real flights) out of it.
    Gaps over `max_gap_s` also split. Flights shorter than `min_samples` are dropped.
    """
    n = len(t)
    if n == 0:
        return []
    w = cfg["window_samples"]
    cuts = {0, n - 1}
    cuts |= {k for k in range(1, n) if t[k] - t[k - 1] > cfg["max_gap_s"]}
    cuts |= {k - 1 for k in range(1, n) if t[k] - t[k - 1] > cfg["max_gap_s"]}
    score = np.zeros(n)
    for k in range(w, n - w):
        if any(t[a + 1] - t[a] > cfg["max_gap_s"] for a in range(k - w, k + w)):
            continue
        # How much splitting at k reduces the squared error, per sample (an F-test in px).
        joint = _sse(t[k - w:k + w + 1], uv[k - w:k + w + 1])
        apart = _sse(t[k - w:k + 1], uv[k - w:k + 1]) + _sse(t[k:k + w + 1], uv[k:k + w + 1])
        score[k] = np.sqrt(max(0.0, joint - apart) / (2 * w + 1))
    # One change point per touch: candidates closer than the shortest flight are thinned
    # to the strongest, so no short weak scraps are left between them.
    gap = cfg["min_samples"] - 1
    for k in np.argsort(-score):
        if score[k] <= cfg["break_px"]:
            break
        if all(abs(k - j) >= gap for j in cuts if 0 < j < n - 1):
            cuts.add(int(k))
    edges = sorted(cuts)
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b - a + 1 >= cfg["min_samples"] and t[b] - t[a] > 0:
            if not any(t[x + 1] - t[x] > cfg["max_gap_s"] for x in range(a, b)):
                out.append((a, b))
    return out


def _sse(t: np.ndarray, uv: np.ndarray) -> float:
    """Summed squared pixel residual of a quadratic-in-time fit."""
    tt = t - t[0]
    A = np.stack([np.ones_like(tt), tt, tt**2], axis=1)
    coef, *_ = np.linalg.lstsq(A, uv, rcond=None)
    return float(((A @ coef - uv) ** 2).sum())


class _Projector:
    """Project point i through camera i, all at once (cameras precomputed)."""

    def __init__(self, cameras: list[Camera]):
        self.R = np.stack([c.R for c in cameras])
        self.t = np.stack([c.tvec for c in cameras])
        self.f = np.array([c.f for c in cameras])
        self.c = np.stack([c.principal_point for c in cameras])
        self.k1 = np.array([c.k1 for c in cameras])

    def __call__(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pc = np.einsum("nij,nj->ni", self.R, pts * np.array([-1.0, 1.0, 1.0])) + self.t
        z = pc[:, 2]
        xy = pc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
        xy = xy * (1 + self.k1 * (xy**2).sum(axis=1))[:, None]
        return xy * self.f[:, None] + self.c, z


def fit_arc(t: np.ndarray, uv: np.ndarray, cameras: list[Camera], g: np.ndarray,
            cfg: dict) -> tuple[list[float], list[float], float, float]:
    """-> (p0, v0, rms_px, depth spread m) for one flight; t relative to its first sample."""
    t = np.asarray(t, float) - t[0]
    T = t[-1]
    r_first = cameras[0].rays(uv[:1])[0]
    r_last = cameras[-1].rays(uv[-1:])[0]
    c_first, c_last = cameras[0].centre, cameras[-1].centre

    proj = _Projector(cameras)

    def residuals(x):
        p0, v0 = x[:3], x[3:]
        pts = p0 + v0 * t[:, None] + 0.5 * g * t[:, None] ** 2
        uv_hat, z = proj(pts)
        r = np.where((z > 0.1)[:, None], uv_hat - uv, 1e3)
        return r.ravel() / cfg["pixel_sigma_px"]

    depths = np.linspace(*cfg["depth_range_m"], cfg["depth_steps"])
    starts = []
    for d0 in depths:
        for d1 in depths:
            p0 = c_first + d0 * r_first
            p1 = c_last + d1 * r_last
            v0 = (p1 - p0) / T - 0.5 * g * T
            x = np.concatenate([p0, v0])
            starts.append((float((residuals(x) ** 2).sum()), x))
    starts.sort(key=lambda s: s[0])
    lo_xy, hi_xy = cfg["bounds_xy_m"]
    vmax = cfg["max_speed_mps"]
    lo = np.array([lo_xy, lo_xy, -cfg["floor_tolerance_m"], -vmax, -vmax, -vmax])
    hi = np.array([hi_xy, hi_xy, cfg["max_height_m"], vmax, vmax, vmax])
    fits = []
    for _, x0 in starts[:cfg["refine_starts"]]:
        sol = least_squares(residuals, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi),
                            loss="soft_l1", f_scale=2.0)
        fits.append((float(sol.cost), sol.x))
    fits.sort(key=lambda f: f[0])
    best = fits[0][1]
    # Depth spread among fits nearly as good as the best: large = depth poorly fixed.
    good = [x for c, x in fits if c <= fits[0][0] * 1.5 + 1.0]
    spread = float(np.ptp([np.linalg.norm(x[:3] - c_first) for x in good])) if good else 0.0
    rms = float(np.sqrt((residuals(best) ** 2).reshape(-1, 2).sum(axis=1).mean())
                * cfg["pixel_sigma_px"])
    return best[:3].tolist(), best[3:].tolist(), rms, spread


def fit_flights(times: np.ndarray, uv_ref: np.ndarray, uv: np.ndarray, valid: np.ndarray,
                cameras: list[Camera], size_px: np.ndarray, cfg: dict) -> list[Arc]:
    """Arcs for every flight in a path; indices refer to the full path arrays."""
    g = np.array([0.0, 0.0, -cfg["gravity"]])
    idx = np.flatnonzero(valid)
    arcs = []
    for i, j in segment(times[idx], uv_ref[idx], cfg["arcs"]):
        sel = idx[i:j + 1]
        p0, v0, rms, spread = fit_arc(times[sel], uv[sel], [cameras[k] for k in sel], g,
                                      cfg["arcs"])
        arc = Arc(int(sel[0]), int(sel[-1]), float(times[sel[0]]), p0, v0, rms,
                  float(times[sel[-1]] - times[sel[0]]), len(sel), float("nan"), 0.0)
        pts = arc.position(times[sel], g)
        dist = np.linalg.norm(pts - np.array([c.centre for c in (cameras[k] for k in sel)]),
                              axis=1)
        expected = np.array([cameras[k].f for k in sel]) * BALL_DIAMETER_M / dist
        ok = np.isfinite(size_px[sel])
        if ok.any():
            arc.size_ratio = float(np.median(size_px[sel][ok] / expected[ok]))
        a = cfg["arcs"]
        z_ok = bool((pts[:, 2] > -a["floor_tolerance_m"]).all()
                    and (pts[:, 2] < a["max_height_m"]).all())
        arc.confidence = float(np.clip(1 - rms / a["max_rms_px"], 0, 1)
                               * min(1.0, arc.duration_s / a["full_confidence_s"])
                               * np.exp(-spread / a["depth_spread_scale_m"])
                               * (1.0 if z_ok else 0.2))
        arcs.append(arc)
    return arcs
