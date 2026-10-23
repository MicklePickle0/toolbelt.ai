"""Automatic calibration from painted lines.

1. Line segments from the line mask (Hough).
2. Vanishing points (RANSAC). Two of them are the court's floor directions:
   transverse lines (end, attack, centre lines, net) and sidelines.
3. For each pair of floor directions and candidate focal length the rotation
   is fixed; the translation follows linearly once three detected lines are
   labelled (two sidelines + one transverse line, or one sideline + two
   transverse lines). Every labelling is tried and scored by how many
   projected court-line samples land on the mask.
4. The best few are refined on the mask (pose, focal length, optionally k1)
   and the most confident one wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import cv2
import numpy as np

from setter_eyes.calibration.camera import Camera, to_cv
from setter_eyes.calibration.court_model import CourtModel
from setter_eyes.calibration.fit import focal_from_hfov, refine_on_lines
from setter_eyes.calibration.lines import LineScorer, Support


@dataclass
class AutoResult:
    camera: Camera
    support: Support
    hypotheses: int
    vanishing_points: int


def detect_segments(mask: np.ndarray, cfg: dict) -> np.ndarray:
    h = cfg["hough"]
    segs = cv2.HoughLinesP(mask, 1, np.pi / 720, h["threshold"],
                           minLineLength=h["min_length_px"], maxLineGap=h["max_gap_px"])
    return np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4).astype(float)


def _homog_lines(segs: np.ndarray) -> np.ndarray:
    ones = np.ones((len(segs), 1))
    lines = np.cross(np.hstack([segs[:, :2], ones]), np.hstack([segs[:, 2:], ones]))
    return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)


def _consistent(vps: np.ndarray, mids: np.ndarray, dirs: np.ndarray, sin_tol: float):
    """(V, N) bool: segment n points at vanishing point v (within the angle tolerance)."""
    d = vps[:, None, :2] - mids[None] * vps[:, None, 2:3]
    d /= np.linalg.norm(d, axis=2, keepdims=True) + 1e-12
    return np.abs(d[..., 0] * dirs[None, :, 1] - d[..., 1] * dirs[None, :, 0]) < sin_tol


def find_vanishing_points(segs: np.ndarray, cfg: dict,
                          rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """-> [(vanishing point (3,) homogeneous, inlier mask over segs)], strongest first."""
    vcfg = cfg["vanishing"]
    if len(segs) < 2:
        return []
    lines = _homog_lines(segs)
    mids = (segs[:, :2] + segs[:, 2:]) / 2
    dirs = segs[:, 2:] - segs[:, :2]
    lengths = np.linalg.norm(dirs, axis=1)
    dirs /= lengths[:, None]
    sin_tol = np.sin(np.radians(vcfg["inlier_angle_deg"]))

    out = []
    remaining = np.ones(len(segs), bool)
    for _ in range(vcfg["max_points"]):
        idx = np.flatnonzero(remaining)
        if len(idx) < 2:
            break
        p = lengths[idx] / lengths[idx].sum()
        pairs = np.array([rng.choice(idx, 2, replace=False, p=p)
                          for _ in range(vcfg["ransac_iterations"])])
        cands = np.cross(lines[pairs[:, 0]], lines[pairs[:, 1]])
        norms = np.linalg.norm(cands, axis=1)
        cands = cands[norms > 1e-12] / norms[norms > 1e-12, None]
        if not len(cands):
            break
        cons = _consistent(cands, mids[idx], dirs[idx], sin_tol)
        best = cands[np.argmax((cons * lengths[idx]).sum(axis=1))]
        inl = idx[_consistent(best[None], mids[idx], dirs[idx], sin_tol)[0]]
        if len(inl) < 2:
            break
        # Least squares refit: v minimising sum w (l . v)^2, |v| = 1.
        weighted = lines[inl] * np.sqrt(lengths[inl])[:, None]
        v = np.linalg.svd(weighted)[2][-1]
        inl_mask = np.zeros(len(segs), bool)
        inl_mask[idx] = _consistent(v[None], mids[idx], dirs[idx], sin_tol)[0]
        if inl_mask.sum() < 2:
            break
        out.append((v, inl_mask))
        remaining &= ~inl_mask
    return out


def family_lines(segs: np.ndarray, inliers: np.ndarray, vp: np.ndarray, k: int,
                 merge_tol: float, centre: np.ndarray) -> np.ndarray:
    """Merge a vanishing point's segments into distinct lines through it -> (<=k, 3), strongest
    first. Neighbouring segments of one painted line (its two edges, broken pieces) merge."""
    s = segs[inliers]
    mids = (s[:, :2] + s[:, 2:]) / 2
    lengths = np.linalg.norm(s[:, 2:] - s[:, :2], axis=1)
    finite = abs(vp[2]) > 1e-9 and np.linalg.norm(vp[:2] / vp[2]) < 1e7
    if finite:
        p = vp[:2] / vp[2]
        key = np.arctan2(mids[:, 1] - p[1], mids[:, 0] - p[0]) % np.pi
        key *= np.linalg.norm(centre - p)          # ~ pixels near the image centre
    else:
        d = vp[:2] / np.linalg.norm(vp[:2])
        key = mids @ np.array([-d[1], d[0]])
    order = np.argsort(key)
    clusters, current = [], [order[0]]
    for a, b in zip(order[:-1], order[1:], strict=True):
        if key[b] - key[a] <= merge_tol:
            current.append(b)
        else:
            clusters.append(current)
            current = [b]
    clusters.append(current)

    out = []
    for c in clusters:
        w = lengths[c]
        m = (mids[c] * w[:, None]).sum(axis=0) / w.sum()
        if finite:
            line = np.cross([m[0], m[1], 1.0], vp)
        else:
            line = np.cross([m[0], m[1], 1.0], [m[0] + vp[0], m[1] + vp[1], 1.0])
        out.append((w.sum(), line / np.linalg.norm(line[:2])))
    out.sort(key=lambda x: -x[0])
    return np.array([line for _, line in out[:k]])


def _focal_candidates(vps: list[np.ndarray], width: int, centre: np.ndarray,
                      cfg: dict) -> np.ndarray:
    lo, hi = cfg["hfov_range_deg"]
    fs = [focal_from_hfov(width, h) for h in np.linspace(lo, hi, cfg["focal_candidates"])]
    f_lo, f_hi = focal_from_hfov(width, hi), focal_from_hfov(width, lo)
    finite = [v[:2] / v[2] for v in vps if abs(v[2]) > 1e-9]
    for a, b in combinations(finite, 2):
        f2 = -np.dot(a - centre, b - centre)
        if f2 > 0 and f_lo <= np.sqrt(f2) <= f_hi:
            fs.append(np.sqrt(f2))
    return np.array(sorted(fs))


def _rotation(vx: np.ndarray, vy: np.ndarray, K: np.ndarray, max_ortho: float):
    """Rotation (mirrored court frame -> camera) whose x/y axes point at vx/vy, or None."""
    Kinv = np.linalg.inv(K)
    dx = Kinv @ vx
    dy = Kinv @ vy
    dx /= np.linalg.norm(dx)
    dy /= np.linalg.norm(dy)
    if abs(dx @ dy) > max_ortho:
        return None
    # Court +x is image-right, so mirrored x points image-left; court y points
    # at the camera (negative camera z); court z is up (negative camera y).
    r1 = dx * -np.sign(dx[0])
    r2 = dy * -np.sign(dy[2])
    r2 = r2 - (r1 @ r2) * r1
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    if r3[1] >= 0:
        return None
    return np.column_stack([r1, r2, r3])


def plausible(cam: Camera, model: CourtModel, cfg: dict) -> bool:
    """Camera placement check: above the floor, on our side, not absurdly far."""
    c = cam.centre
    hmin, hmax = cfg["camera_height_m"]
    mid = np.array([model.lines["centre"][:, 0].mean(), 0.0, 0.0])
    return bool(hmin <= c[2] <= hmax and c[1] >= cfg["camera_min_y_m"]
                and np.linalg.norm(c - mid) <= cfg["camera_max_distance_m"]
                and cam.project(mid[None])[1][0] > 0)


def _image_row(line: np.ndarray, x: float) -> float:
    return -(line[0] * x + line[2]) / line[1] if abs(line[1]) > 1e-9 else np.inf


def _hypotheses(LX: np.ndarray, LY: np.ndarray, a: np.ndarray, b: np.ndarray,
                xs: list[float], ys: list[float], cx: float):
    """Linear systems for every labelling -> (A (H,3,3), rhs (H,3)).

    Floor point (X', Y) maps to H [X', Y, 1] with H = [a b t']; a line l through
    image points of X' = c satisfies t'.l = -c (a.l); of Y = c, t'.l = -c (b.l).
    """
    rows, rhs = [], []
    # Two sidelines (either order) + one transverse line.
    for i, j in permutations(range(len(LY)), 2):
        for k in range(len(LX)):
            for y in ys:
                rows.append([LY[i], LY[j], LX[k]])
                rhs.append([-xs[0] * (a @ LY[i]), -xs[1] * (a @ LY[j]), -y * (b @ LX[k])])
    # One sideline + two transverse lines; the higher one in the image is farther away.
    order = sorted(range(len(LX)), key=lambda k: _image_row(LX[k], cx))
    for i in range(len(LY)):
        for x in xs:
            for k, m in combinations(order, 2):
                for y1, y2 in combinations(sorted(ys), 2):
                    rows.append([LY[i], LX[k], LX[m]])
                    rhs.append([-x * (a @ LY[i]), -y1 * (b @ LX[k]), -y2 * (b @ LX[m])])
    if not rows:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.array(rows), np.array(rhs)


def auto_calibrate(image_bgr: np.ndarray, model: CourtModel, cfg: dict,
                   mask: np.ndarray | None = None, scale: float | None = None,
                   seed: int = 0) -> AutoResult | None:
    """Calibrate from one (ideally player-free) frame. None if no candidate is plausible."""
    from setter_eyes.calibration.lines import line_mask

    height, width = image_bgr.shape[:2]
    if mask is None:
        mask, scale = line_mask(image_bgr, cfg)
    scorer = LineScorer(mask, scale, model, cfg)
    search = LineScorer(mask, scale, model, cfg, step=cfg["search_sample_step_m"])
    segs_small = detect_segments(mask, cfg)
    vps_small = find_vanishing_points(segs_small, cfg, np.random.default_rng(seed))
    if len(vps_small) < 2:
        return None

    # Everything below is in full-resolution pixels.
    segs = segs_small / scale
    vps = [np.array([v[0] / scale, v[1] / scale, v[2]]) for v, _ in vps_small]
    centre = np.array([width / 2, height / 2])
    k = cfg["lines_per_family"]
    families = [family_lines(segs, inl, vp, k, cfg["merge_tolerance_px"] / scale, centre)
                for vp, (_, inl) in zip(vps, vps_small, strict=True)]

    lines = model.lines
    xs = [to_cv(lines["side_left"][:1])[0, 0], to_cv(lines["side_right"][:1])[0, 0]]
    ys = [lines[n][0, 1] for n in ("end_near", "attack_near", "centre", "attack_far", "end_far")]
    pts = to_cv(search.pts)
    hmin, hmax = cfg["camera_height_m"]
    court_mid = np.array([model.lines["centre"][:, 0].mean(), 0.0, 0.0])

    kept: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    n_hyp = 0
    for ix, iy in permutations(range(len(vps)), 2):
        LX, LY = families[ix], families[iy]
        if len(LX) < 1 or len(LY) < 1:
            continue
        for f in _focal_candidates(vps, width, centre, cfg):
            K = np.array([[f, 0, centre[0]], [0, f, centre[1]], [0, 0, 1.0]])
            R = _rotation(vps[ix], vps[iy], K, cfg["vanishing"]["max_orthogonality_error"])
            if R is None:
                continue
            A, rhs = _hypotheses(LX, LY, K @ R[:, 0], K @ R[:, 1], xs, ys, centre[0])
            ok = np.abs(np.linalg.det(A)) > 1e-12
            if not ok.any():
                continue
            t = np.linalg.solve(A[ok], rhs[ok][..., None])[..., 0] @ np.linalg.inv(K).T
            n_hyp += len(t)
            C = (-(t @ R)) * np.array([-1.0, 1.0, 1.0])          # camera centres, court frame
            mid_depth = (R @ to_cv(court_mid[None])[0])[2] + t[:, 2]
            placed = ((C[:, 2] >= hmin) & (C[:, 2] <= hmax) &
                         (C[:, 1] >= cfg["camera_min_y_m"]) & (mid_depth > 0) &
                         (np.linalg.norm(C - court_mid, axis=1) <= cfg["camera_max_distance_m"]))
            t = t[placed]
            if not len(t):
                continue
            pc = pts @ R.T
            for chunk in np.array_split(t, max(1, len(t) // 1500)):
                P = pc[None] + chunk[:, None, :]
                z = P[..., 2]
                uv = f * P[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[..., None] + centre
                d, vis = search.lookup(uv, z)
                score = (vis & (d <= search.tol)).sum(axis=1)
                for s_i in np.argsort(-score)[:3]:
                    kept.append((int(score[s_i]), float(f), R, chunk[s_i]))

    if not kept:
        return None
    kept.sort(key=lambda x: -x[0])
    chosen: list[Camera] = []
    for _, f, R, t in kept:
        cam = Camera(width, height, f, cv2.Rodrigues(R)[0], t)
        if any(np.linalg.norm(cam.centre - c.centre) < 0.3 and abs(cam.f / c.f - 1) < 0.03
               for c in chosen):
            continue
        chosen.append(cam)
        if len(chosen) >= cfg["refine_top_k"]:
            break

    best: tuple[Camera, Support] | None = None
    for cam in chosen:
        cam = refine_on_lines(cam, scorer, cfg)
        if not plausible(cam, model, cfg):
            continue
        sup = scorer.evaluate(cam)
        if best is None or (sup.confidence, sup.inliers) > (best[1].confidence, best[1].inliers):
            best = (cam, sup)
    if best is None:
        return None
    cam, sup = best
    if cfg.get("estimate_distortion"):
        cam_k1 = refine_on_lines(cam, scorer, cfg, estimate_k1=True)
        sup_k1 = scorer.evaluate(cam_k1)
        if plausible(cam_k1, model, cfg) and sup_k1.confidence >= sup.confidence:
            cam, sup = cam_k1, sup_k1
    return AutoResult(cam, sup, n_hyp, len(vps))
