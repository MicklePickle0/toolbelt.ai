"""Follow a panning camera: one rotation per frame about the calibrated, fixed camera centre.

The Humber footage pans with play while the camera stays put, so each frame's pose is
the calibrated camera rotated. Each frame is tracked against a KEYFRAME, not against
the previous frame: per-frame pans are tiny (~0.001 deg), and chaining thousands of
noisy frame-to-frame steps random-walks off by degrees.

1. LK optical flow from the keyframe's corners to the current frame (initialised from
   the previous frame's positions); a homography between undistorted rays is a pure
   rotation, and players are RANSAC outliers.
2. When tracking weakens or the camera has panned `keyframe_spacing_deg` from the
   keyframe, switch: first try stored keyframes near the current angle (ORB-verified,
   so returning to an old view re-uses it and adds no drift), else make the current
   frame a new keyframe.
3. At the end, rotations are re-expressed relative to the calibrated frame.

Confidence per frame = RANSAC inlier share x share of corners still tracked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from setter_eyes.calibration.camera import Camera, to_cv
from setter_eyes.calibration.store import Calibration
from setter_eyes.ingest import Frame, iter_frames_scaled


@dataclass
class PoseTrack:
    video: str
    base: Camera                 # the calibrated camera (centre, focal length, k1)
    ref_frame: int
    frames: np.ndarray           # (N,) frame indices
    rvecs: np.ndarray            # (N, 3) absolute rotations
    confidence: np.ndarray       # (N,)

    def camera(self, frame: int) -> Camera:
        i = int(np.clip(np.searchsorted(self.frames, frame), 0, len(self.frames) - 1))
        if self.frames[i] != frame and i > 0 and \
                abs(self.frames[i - 1] - frame) < abs(self.frames[i] - frame):
            i -= 1
        R = cv2.Rodrigues(self.rvecs[i])[0]
        t = -R @ to_cv(self.base.centre[None])[0]
        b = self.base
        return Camera(b.width, b.height, b.f, self.rvecs[i], t, b.k1)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        b = self.base
        np.savez_compressed(path, video=self.video, ref_frame=self.ref_frame, frames=self.frames,
                            rvecs=self.rvecs, confidence=self.confidence,
                            base=np.array([b.width, b.height, b.f, *b.rvec, *b.tvec, b.k1]))
        return path

    @classmethod
    def load(cls, path: Path) -> PoseTrack:
        d = np.load(path)
        b = d["base"]
        base = Camera(int(b[0]), int(b[1]), float(b[2]), b[3:6], b[6:9], float(b[9]))
        return cls(str(d["video"]), base, int(d["ref_frame"]), d["frames"], d["rvecs"],
                   d["confidence"])


def _polar(H: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    return R if np.linalg.det(R) > 0 else -R


def rotation_between(rays_a: np.ndarray, rays_b: np.ndarray, f: float,
                     thresh_px: float) -> tuple[np.ndarray, float] | None:
    """Rotation R with rays_b ~ R rays_a (normalised image coordinates, (N, 2)) and the
    inlier share, or None. f converts the pixel threshold to ray units."""
    if len(rays_a) < 8:
        return None
    H, inl = cv2.findHomography(rays_a.astype(np.float32), rays_b.astype(np.float32),
                                cv2.RANSAC, thresh_px / f)
    if H is None:
        return None
    s = np.linalg.svd(H, compute_uv=False)
    if s[0] / s[2] > 1.15:          # not a rotation: zoom, translation or a bad fit
        return None
    return _polar(H), float(inl.mean())


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra @ Rb.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


@dataclass
class _Keyframe:
    R: np.ndarray                # rotation relative to the first frame
    gray: np.ndarray
    pts: np.ndarray              # (N, 1, 2) float32 corners
    rays: np.ndarray             # (N, 2) undistorted corners
    kp_rays: np.ndarray          # ORB keypoints, undistorted
    des: np.ndarray | None


class _Tracker:
    def __init__(self, base: Camera, cfg: dict):
        self.t = cfg["track"]
        s = self.t["width"] / base.width
        self.scale = s
        self.K = base.K * np.array([[s], [s], [1.0]])
        self.dist = np.array([base.k1, 0.0, 0.0, 0.0])
        self.f = base.f * s
        self.orb = cv2.ORB_create(self.t["orb_features"])
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.library: list[_Keyframe] = []

    def rays(self, pts: np.ndarray) -> np.ndarray:
        if not len(pts):
            return np.zeros((0, 2))
        return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), self.K,
                                   self.dist).reshape(-1, 2)

    def project(self, r: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Rays rotated by R -> pixels (N, 1, 2) float32."""
        pts3 = np.hstack([r, np.ones((len(r), 1))])
        uv, _ = cv2.projectPoints(pts3, cv2.Rodrigues(R)[0], np.zeros(3), self.K, self.dist)
        return uv.astype(np.float32)

    def keyframe(self, gray: np.ndarray, R: np.ndarray) -> _Keyframe | None:
        t = self.t
        pts = cv2.goodFeaturesToTrack(gray, t["max_corners"], 0.01, t["min_corner_distance"])
        if pts is None or len(pts) < 8:
            return None
        kps, des = self.orb.detectAndCompute(gray, None)
        kp_rays = self.rays(np.array([k.pt for k in kps])) if kps else np.zeros((0, 2))
        return _Keyframe(R.copy(), gray, pts, self.rays(pts), kp_rays, des)

    def relocalise(self, gray: np.ndarray, R_now: np.ndarray,
                   exclude: _Keyframe) -> tuple[_Keyframe, np.ndarray] | None:
        """A stored keyframe that ORB-matches the frame near R_now -> (keyframe, R), or None."""
        t = self.t
        cands = sorted((k for k in self.library if k is not exclude and k.des is not None
                        and _angle_deg(k.R, R_now) <= 2 * t["keyframe_spacing_deg"]),
                       key=lambda k: _angle_deg(k.R, R_now))[:3]
        if not cands:
            return None
        kps, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kps) < 8:
            return None
        cur = self.rays(np.array([k.pt for k in kps]))
        for kf in cands:
            matches = self.matcher.match(kf.des, des)
            if len(matches) < t["min_matches"]:
                continue
            res = rotation_between(kf.kp_rays[[m.queryIdx for m in matches]],
                                   cur[[m.trainIdx for m in matches]], self.f, t["ransac_px"])
            if res is None or res[1] < t["min_keyframe_inliers"]:
                continue
            R = res[0] @ kf.R
            # Repetitive texture (curtain panels, bleachers) can match one period off.
            if _angle_deg(R, R_now) <= t["max_keyframe_jump_deg"]:
                return kf, R
        return None


def track_rotation(frames: Iterable[Frame], cal: Calibration, cfg: dict,
                   progress: Callable[[int], None] | None = None,
                   on_frame: Callable[[Frame, np.ndarray, float], None] | None = None
                   ) -> PoseTrack:
    """Track every frame of `frames` (in order, starting at 0) -> absolute poses.

    on_frame(frame, R_relative_to_frame0, confidence) is called per frame (for previews
    that don't need absolute poses yet; use PoseTrack afterwards for those).
    """
    base = cal.camera
    tr = _Tracker(base, cfg)
    t = tr.t
    kf: _Keyframe | None = None
    pred: np.ndarray | None = None          # keyframe corners' positions in the last frame
    R_cum = np.eye(3)
    R_ref = None
    idx_list, R_list, conf_list = [], [], []

    for frame in frames:
        img = frame.image
        if img.shape[1] != t["width"]:
            img = cv2.resize(img, None, fx=tr.scale, fy=tr.scale, interpolation=cv2.INTER_AREA)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        conf = 0.0
        if kf is None:
            kf = tr.keyframe(gray, R_cum)
            if kf is not None:
                tr.library.append(kf)
                pred, conf = kf.pts.copy(), 1.0
        else:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(kf.gray, gray, kf.pts, pred.copy(),
                                                  winSize=(21, 21), maxLevel=4,
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            ok = st.ravel() == 1
            res = rotation_between(kf.rays[ok], tr.rays(nxt[ok]), tr.f, t["ransac_px"])
            if res is not None:
                R_cum = res[0] @ kf.R
                conf = res[1] * ok.mean()
            if conf < t["min_flow_confidence"] or \
                    _angle_deg(R_cum, kf.R) > t["keyframe_spacing_deg"]:
                found = tr.relocalise(gray, R_cum, kf)
                if found is not None:
                    kf, R_cum = found
                    conf = max(conf, t["min_flow_confidence"])
                else:
                    new = tr.keyframe(gray, R_cum)
                    if new is not None and (conf >= t["min_flow_confidence"] or res is None):
                        # A healthy new view joins the library; after losing track we
                        # restart from here but don't trust it as a reference later.
                        if res is not None:
                            tr.library.append(new)
                        kf = new
            pred = tr.project(kf.rays, R_cum @ kf.R.T)

        if frame.index == cal.frame:
            R_ref = R_cum.copy()
        idx_list.append(frame.index)
        R_list.append(R_cum.copy())
        conf_list.append(conf)
        if on_frame is not None:
            on_frame(frame, R_cum, conf)
        if progress is not None and len(idx_list) % 1000 == 0:
            progress(frame.index)

    if R_ref is None:
        raise ValueError(f"calibration frame {cal.frame} was not among the tracked frames")
    # R_abs(t) = R_cum(t) R_cum(ref)^T R_base: rotate the calibrated pose by the pan since ref.
    to_abs = R_ref.T @ base.R
    rvecs = np.array([cv2.Rodrigues(R @ to_abs)[0].ravel() for R in R_list])
    return PoseTrack(cal.video, base, int(cal.frame), np.array(idx_list), rvecs,
                     np.array(conf_list))


def track_video(path: Path, cal: Calibration, cfg: dict, **kwargs) -> PoseTrack:
    """Track a whole video, decoded straight at the tracking width (BGR, for previews)."""
    return track_rotation(iter_frames_scaled(path, cfg["track"]["width"]), cal, cfg, **kwargs)
