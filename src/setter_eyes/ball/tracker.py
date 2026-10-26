"""Ball tracking over a whole video: detections -> tracks -> the match ball's path.

1. Pan removal: each detection's pixel is re-expressed in the calibrated reference
   view (same camera centre, the reference orientation), so camera panning doesn't
   look like ball motion.
2. Tracks: one Kalman filter (kalman.py) per candidate; detections join the nearest
   track within its gate (greedy, by normalised distance); strong unmatched detections
   start tentative tracks; tracks die after `max_misses` samples without a detection.
3. Match ball: per sample, the live track with the best quality: long, confident and
   moving. Tracks that never move (spare balls on the floor, marks on the wall) are
   never chosen. Short gaps (<= `bridge_samples`) are filled by smoothing between the
   samples on either side (offline, so both sides are known).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from setter_eyes.ball.kalman import BallKalman
from setter_eyes.calibration.camera import Camera


@dataclass
class Detections:
    frames: np.ndarray        # every processed frame
    det_frame: np.ndarray
    det_box: np.ndarray       # (N, 4) x1, y1, x2, y2
    det_score: np.ndarray
    fps: float

    @classmethod
    def load(cls, path) -> Detections:
        d = np.load(path)
        return cls(d["frames"], d["det_frame"], d["det_box"], d["det_score"], float(d["fps"]))

    def by_frame(self) -> dict[int, np.ndarray]:
        """frame -> indices into det_* arrays."""
        out: dict[int, list[int]] = {}
        for i, f in enumerate(self.det_frame):
            out.setdefault(int(f), []).append(i)
        return {f: np.array(v) for f, v in out.items()}


def stabilise(cameras, ref: Camera, uv: np.ndarray) -> np.ndarray:
    """Pixels seen by cameras[i] -> pixels the reference camera would see (no distortion).
    Exact for a camera that only rotates about its centre."""
    out = np.empty((len(uv), 2))
    for i, (cam, p) in enumerate(zip(cameras, uv, strict=True)):
        d = cam.rays(p[None])[0]                          # court-frame direction
        pc = ref.R @ (d * np.array([-1.0, 1.0, 1.0]))     # into the reference camera
        out[i] = ref.f * pc[:2] / pc[2] + ref.principal_point
    return out


def unstabilise(cam: Camera, ref: Camera, uv_ref: np.ndarray) -> np.ndarray:
    """Inverse of stabilise for one camera: reference-view pixels -> cam's pixels."""
    xy = (np.asarray(uv_ref, float).reshape(-1, 2) - ref.principal_point) / ref.f
    d_ref = np.hstack([xy, np.ones((len(xy), 1))])
    d_court = (d_ref @ ref.R) * np.array([-1.0, 1.0, 1.0])
    pts = cam.centre + d_court * 10.0                     # any point along the ray
    return cam.project(pts)[0]


@dataclass
class Track:
    id: int
    kf: BallKalman
    hits: int = 1
    misses: int = 0
    samples: list[tuple[int, float, float, float, int]] = field(default_factory=list)
    # (frame, u_ref, v_ref, score, detection index) for measured samples

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3


@dataclass
class BallPath:
    """The match ball, one row per processed frame (NaN where unknown)."""
    frames: np.ndarray
    track_id: np.ndarray      # -1 = none
    uv_ref: np.ndarray        # (N, 2) pan-free reference-view pixels
    uv: np.ndarray            # (N, 2) image pixels of that frame
    measured: np.ndarray      # bool: a detection (else bridged by prediction)
    score: np.ndarray
    size_px: np.ndarray       # detection box size (NaN when bridged)

    def save(self, path) -> None:
        np.savez_compressed(path, frames=self.frames, track_id=self.track_id,
                            uv_ref=self.uv_ref, uv=self.uv, measured=self.measured,
                            score=self.score, size_px=self.size_px)

    @classmethod
    def load(cls, path) -> BallPath:
        d = np.load(path)
        return cls(d["frames"], d["track_id"], d["uv_ref"], d["uv"], d["measured"],
                   d["score"], d["size_px"])


def track_ball(dets: Detections, camera_at, ref: Camera, cfg: dict) -> BallPath:
    """camera_at(frame) -> Camera for that frame (pan included)."""
    fps = dets.fps
    gravity_px = ref.f * cfg["gravity"] / cfg["typical_depth_m"]
    groups = dets.by_frame()
    centres = (dets.det_box[:, :2] + dets.det_box[:, 2:]) / 2
    sizes = (dets.det_box[:, 2:] - dets.det_box[:, :2]).mean(axis=1)
    # Pan-free coordinates for every detection, once.
    uv_ref_all = np.empty_like(centres)
    for f, idx in groups.items():
        cam = camera_at(f)
        uv_ref_all[idx] = stabilise([cam] * len(idx), ref, centres[idx])

    tracks: list[Track] = []
    finished: list[Track] = []
    next_id = 0
    # Detection may skip the time between rallies (play spans): no track lives across a
    # skipped stretch longer than a track may go unseen.
    step = int(np.median(np.diff(dets.frames))) if len(dets.frames) > 1 else 1
    prev_f = None
    for f in dets.frames:
        f = int(f)
        t = f / fps
        if prev_f is not None and (f - prev_f) / step - 1 > cfg["max_misses"]:
            finished += tracks
            tracks = []
        prev_f = f
        idx = groups.get(f, np.array([], int))
        # Gate every (track, detection) pair; greedy assignment by normalised distance.
        pairs = []
        for ti, tr in enumerate(tracks):
            pos, P = tr.kf.peek(t)
            radius = cfg["gate_px"] + 3 * np.sqrt(max(P[0, 0], P[1, 1]))
            for di in idx:
                dist = np.linalg.norm(uv_ref_all[di] - pos)
                if dist <= radius:
                    pairs.append((dist / radius, ti, int(di)))
        used_t, used_d = set(), set()
        for _, ti, di in sorted(pairs):
            if ti in used_t or di in used_d:
                continue
            tr = tracks[ti]
            tr.kf.predict(t)
            sigma = max(cfg["meas_sigma_px"], cfg["meas_sigma_rel"] * sizes[di])
            tr.kf.update(uv_ref_all[di], sigma)
            tr.hits += 1
            tr.misses = 0
            tr.samples.append((f, *uv_ref_all[di], float(dets.det_score[di]), di))
            used_t.add(ti)
            used_d.add(di)
        survivors = []
        for ti, tr in enumerate(tracks):
            if ti not in used_t:
                tr.misses += 1
            limit = cfg["max_misses"] if tr.confirmed else cfg["tentative_max_misses"]
            (survivors if tr.misses <= limit else finished).append(tr)
        tracks = survivors
        for di in idx:
            if int(di) not in used_d and dets.det_score[di] >= cfg["birth_score"]:
                tr = Track(next_id, BallKalman(uv_ref_all[di], t, cfg["kalman"], gravity_px))
                tr.samples.append((f, *uv_ref_all[di], float(dets.det_score[di]), int(di)))
                tracks.append(tr)
                next_id += 1
    finished += tracks

    # Offline: a track's whole history counts once it proves to be the ball (including
    # samples from before it was confirmed). Gaps of <= bridge_samples are filled by a
    # quadratic through the neighbouring samples (smoothing, not forward prediction).
    quality = {tr.id: _quality(tr, cfg) for tr in finished}
    k_of = {int(f): k for k, f in enumerate(dets.frames)}
    cover: dict[int, list[tuple[float, bool, int, np.ndarray, int]]] = {}
    for tr in finished:
        q = quality[tr.id]
        if q <= 0:
            continue
        s = sorted(tr.samples)
        ks = np.array([k_of[int(x[0])] for x in s])
        uvs = np.array([x[1:3] for x in s], float)
        for (f, u, v, _, di) in s:
            cover.setdefault(k_of[int(f)], []).append((q, True, tr.id, np.array([u, v]), di))
        for a in range(len(s) - 1):
            missing = range(ks[a] + 1, ks[a + 1])
            if not missing or len(missing) > cfg["bridge_samples"]:
                continue
            w = slice(max(0, a - 2), min(len(s), a + 4))
            tt = dets.frames[ks[w]] / fps
            deg = min(2, len(tt) - 1)
            coef = np.polyfit(tt, uvs[w], deg)
            for k in missing:
                tk = dets.frames[k] / fps
                pos = np.array([np.polyval(coef[:, 0], tk), np.polyval(coef[:, 1], tk)])
                cover.setdefault(k, []).append((q, False, tr.id, pos, -1))

    n = len(dets.frames)
    path = BallPath(dets.frames.copy(), np.full(n, -1), np.full((n, 2), np.nan),
                    np.full((n, 2), np.nan), np.zeros(n, bool), np.full(n, np.nan),
                    np.full(n, np.nan))
    for k, cands in cover.items():
        q, measured, tid, pos, di = max(cands, key=lambda c: (c[0], c[1]))
        path.track_id[k] = tid
        path.uv_ref[k] = pos
        path.uv[k] = unstabilise(camera_at(int(dets.frames[k])), ref, pos)[0]
        path.measured[k] = measured
        if measured:
            path.score[k] = dets.det_score[di]
            path.size_px[k] = sizes[di]
    return path


def _quality(tr: Track, cfg: dict) -> float:
    """Long, confident, moving tracks are the match ball; static ones (spare balls,
    marks on the wall) score near zero."""
    if not tr.confirmed:
        return 0.0
    s = np.array(tr.samples)
    if len(s) < 2:
        return 0.0
    uv, score = s[:, 1:3], s[:, 3]
    spread = float(np.linalg.norm(np.ptp(uv, axis=0)))
    if spread < cfg["static_spread_px"]:
        return 0.0                         # never moves: a spare ball or a mark, not play
    return float(len(s) * score.mean())


def path_cameras(path: BallPath, camera_at) -> list[Camera]:
    return [camera_at(int(f)) for f in path.frames]


def draw_trail(image: np.ndarray, pts: np.ndarray, colour, thickness: int) -> None:
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > 1:
        cv2.polylines(image, [np.round(pts).astype(np.int32)], False, colour, thickness,
                      cv2.LINE_AA)
