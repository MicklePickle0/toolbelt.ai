"""Did the camera move? Check a pan track against the calibration frame.

The pan tracker (calibration/track.py) models a camera that only ROTATES about its
calibrated centre. If someone moves or bumps the tripod, or zooms, rotation can't explain
the new view: near and far things shift by different amounts (parallax), and the court
overlay, player positions and ball 3D drift off by a fixed amount from then on.

Check: ORB-match background features between the calibration frame and a sampled frame,
predict where the calibration frame's features should be from the tracked rotation, and
take the median offset over the matches that agree with one another (RANSAC; players and
wrong matches drop out). A pan-only camera stays within a pixel or two (at the tracking
width); a moved one jumps to tens of pixels and stays there. Where the offset jumps between
two samples, the move is narrowed down by bisection.

With later calibrations (the camera was moved and recalibrated, track.reanchor), each
frame is compared with the calibration frame of its own segment.

Frames are read by seeking, which isn't frame-exact; pans are slow enough (~0.001 deg a
frame) that a few frames off doesn't matter here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from setter_eyes.calibration.store import Calibration
from setter_eyes.calibration.track import PoseTrack, _Tracker


@dataclass
class Sample:
    frame: int
    offset_px: float | None      # median background offset, full-resolution pixels
    matches: int


@dataclass
class Report:
    video: str
    fps: float
    samples: list[Sample] = field(default_factory=list)
    moves: list[tuple[int, int]] = field(default_factory=list)   # (last good, first moved)
    verdict: str = ""            # ok | drifting | moved

    def to_dict(self) -> dict:
        return asdict(self)


class _Probe:
    def __init__(self, video_path, cal: Calibration, track: PoseTrack, cfg: dict):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise OSError(f"cannot open {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = cfg["track"]["width"]
        a = cfg["alignment"]
        self.ransac_px = a["ransac_px"]
        self.min_matches = a["min_matches"]
        self.orb = cv2.ORB_create(a["orb_features"])
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.track = track
        # (start frame, tracker for that camera, calibration frame, its ORB features)
        self.refs = []
        for start, base, ref in [(0, cal.camera, cal.frame)] + \
                [(s.start, s.base, s.ref_frame) for s in track.segments]:
            tr = _Tracker(base, cfg)
            gray = self.grab(ref)
            kp, des = self.orb.detectAndCompute(gray, None) if gray is not None else ([], None)
            rays = tr.rays(np.array([k.pt for k in kp])) if kp else np.zeros((0, 2))
            self.refs.append((start, tr, ref, des, rays))

    def _rvec(self, frame: int) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.track.frames, frame), 0, len(self.track.frames) - 1))
        return self.track.rvecs[i]

    def grab(self, frame: int) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, im = self.cap.read()
        if not ok:
            return None
        h = round(im.shape[0] * self.width / im.shape[1])
        small = cv2.resize(im, (self.width, h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def measure(self, frame: int) -> Sample:
        _, tr, ref, ref_des, ref_rays = [r for r in self.refs if frame >= r[0]][-1]
        gray = self.grab(frame)
        if gray is None or ref_des is None:
            return Sample(frame, None, 0)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None:
            return Sample(frame, None, 0)
        m = self.bf.match(ref_des, des)
        if len(m) < self.min_matches:
            return Sample(frame, None, len(m))
        a = ref_rays[[x.queryIdx for x in m]]
        seen = np.array([kp[x.trainIdx].pt for x in m])
        R = cv2.Rodrigues(self._rvec(frame))[0] @ cv2.Rodrigues(self._rvec(ref))[0].T
        pred = tr.project(a, R).reshape(-1, 2)
        _, inl = cv2.findHomography(pred.astype(np.float32), seen.astype(np.float32),
                                    cv2.RANSAC, self.ransac_px)
        if inl is None or inl.sum() < self.min_matches:
            return Sample(frame, None, len(m))
        off = np.linalg.norm(pred - seen, axis=1)[inl.ravel() > 0]
        return Sample(frame, round(float(np.median(off)) / tr.scale, 1), int(inl.sum()))

    def close(self) -> None:
        self.cap.release()


def check_alignment(video_path, cal: Calibration, track: PoseTrack, cfg: dict,
                    every_s: float | None = None,
                    progress=None) -> Report:
    """Sample the video every `every_s` seconds; flag where the camera moved."""
    a = cfg["alignment"]
    probe = _Probe(video_path, cal, track, cfg)
    try:
        fps = probe.fps
        step = max(1, round((every_s or a["every_s"]) * fps))
        frames = list(range(0, int(track.frames[-1]) + 1, step))
        report = Report(track.video, fps)
        for f in frames:
            report.samples.append(probe.measure(f))
            if progress:
                progress(report.samples[-1])
        moved = a["moved_px"]
        good = [s for s in report.samples if s.offset_px is not None]
        for s0, s1 in zip(good, good[1:], strict=False):
            if s0.offset_px < moved <= s1.offset_px:
                report.moves.append(_bisect(probe, s0.frame, s1.frame, moved,
                                            round(a["bisect_s"] * fps)))
        offs = np.array([s.offset_px for s in good]) if good else np.zeros(0)
        if not len(offs):
            report.verdict = "unknown"
        elif (offs >= moved).mean() > 0:
            report.verdict = "moved"
        elif np.percentile(offs, 90) >= a["drifting_px"]:
            report.verdict = "drifting"
        else:
            report.verdict = "ok"
        return report
    finally:
        probe.close()


def _bisect(probe: _Probe, lo: int, hi: int, moved: float, resolution: int) -> tuple[int, int]:
    """Narrow (good lo, moved hi) down to `resolution` frames."""
    while hi - lo > resolution:
        mid = (lo + hi) // 2
        s = probe.measure(mid)
        if s.offset_px is None:
            break
        if s.offset_px >= moved:
            hi = mid
        else:
            lo = mid
    return lo, hi
