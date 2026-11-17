"""Rough play activity over a video: how much moves on the court, camera pan removed.

Not rally segmentation (Milestone 4). Used to sample labelling frames from live play
instead of the time between points, and to skip the time between rallies in detection
(play_spans). Per sample (every `step_s`): warp the previous sample's frame onto this one
with a background homography (LK + RANSAC, so the pan cancels), and score the share of
pixels that changed in the court band of the image.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from setter_eyes.ingest import iter_frames_scaled, probe


def motion_score(prev: np.ndarray, cur: np.ndarray, cfg: dict) -> float:
    """Share of moving pixels in the court band after cancelling the camera's motion."""
    h, w = cur.shape
    pts = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
    warped, valid = prev, np.full(cur.shape, 255, np.uint8)
    if pts is not None and len(pts) >= 8:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
        ok = st.ravel() == 1
        if ok.sum() >= 8:
            H, _ = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 1.0)
            if H is not None:
                warped = cv2.warpPerspective(prev, H, (w, h))
                valid = cv2.warpPerspective(np.full(prev.shape, 255, np.uint8), H, (w, h))
    diff = cv2.absdiff(cv2.GaussianBlur(cur, (5, 5), 0), cv2.GaussianBlur(warped, (5, 5), 0))
    top, bottom = (round(f * h) for f in cfg["court_band"])
    band = (valid[top:bottom] == 255)
    moving = (diff[top:bottom] >= cfg["diff_threshold"]) & band
    return float(moving.sum() / max(1, band.sum()))


def activity_series(path: Path, cfg: dict,
                    progress: Callable[[int], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """-> (frame indices, smoothed activity), one sample every cfg['step_s'] seconds."""
    step = max(1, round(cfg["step_s"] * probe(path).fps))
    idx, raw = [], []
    prev = None
    for frame in iter_frames_scaled(path, cfg["width"], gray=True):
        if frame.index % step:
            continue
        if prev is not None:
            idx.append(frame.index)
            raw.append(motion_score(prev, frame.image, cfg))
        prev = frame.image
        if progress is not None and len(idx) % 1000 == 0 and idx:
            progress(frame.index)
    raw_arr = np.array(raw)
    k = max(1, cfg["smooth_samples"])
    smooth = np.convolve(raw_arr, np.ones(k) / k, mode="same") if len(raw_arr) else raw_arr
    return np.array(idx), smooth


def choose_bursts(idx: np.ndarray, activity: np.ndarray, frame_count: int, cfg: dict,
                  seed: str) -> list[dict]:
    """Bursts of consecutive frames for labelling: most from active play, the rest from
    downtime, each group spread over the whole video -> [{start, len, kind, activity}]."""
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    n = cfg["bursts"]
    n_active = round(n * cfg["active_share"])
    length = cfg["burst_len"]
    hi = np.quantile(activity, cfg["active_quantile"])
    lo = np.quantile(activity, 1 - cfg["active_quantile"])
    out: list[dict] = []
    for kind, pool, k in (("active", activity >= hi, n_active),
                          ("downtime", activity <= lo, n - n_active)):
        cand = np.flatnonzero(pool)
        if not len(cand) or not k:
            continue
        # One pick per equal stretch of the video, so a burst of long rallies can't dominate.
        for stretch in np.array_split(np.arange(len(idx)), k):
            inside = cand[(cand >= stretch[0]) & (cand <= stretch[-1])] if len(stretch) else []
            i = int(rng.choice(inside if len(inside) else cand))
            start = int(np.clip(idx[i] - length // 2, 0, frame_count - length))
            out.append({"start": start, "len": length, "kind": kind,
                        "activity": round(float(activity[i]), 5)})
    return sorted(out, key=lambda b: b["start"])


def play_spans(idx: np.ndarray, activity: np.ndarray, fps: float, frame_count: int,
               cfg: dict) -> list[tuple[int, int]]:
    """Stretches of play to detect in -> [(first frame, end frame)], end exclusive.

    Active samples are above low + level x (high - low) of this video's own activity
    quantiles. Each run of them is padded (the serve comes before the court moves), runs
    closer than min_gap_s are joined, and joined spans with less than min_span_s of
    activity in them are dropped as noise."""
    if not len(idx):
        return [(0, frame_count)]
    lo, hi = np.quantile(activity, [cfg["low_quantile"], cfg["high_quantile"]])
    on = activity >= lo + cfg["level"] * (hi - lo)
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    merged: list[list[float]] = []            # [start, end, active seconds]
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        first, last = int(idx[a]), int(idx[b - 1])
        start = max(0, first - round(cfg["pad_before_s"] * fps))
        end = min(frame_count, last + round(cfg["pad_after_s"] * fps) + 1)
        active = (last - first) / fps
        if merged and start - merged[-1][1] < cfg["min_gap_s"] * fps:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] += active
        else:
            merged.append([start, end, active])
    return [(int(s), int(e)) for s, e, active in merged if active >= cfg["min_span_s"]]


def in_spans(spans: list[tuple[int, int]]) -> Callable[[int], bool]:
    """frame -> inside one of the (sorted, non-overlapping) spans."""
    starts = np.array([s for s, _ in spans])
    ends = np.array([e for _, e in spans])

    def inside(frame: int) -> bool:
        i = int(np.searchsorted(starts, frame, side="right")) - 1
        return i >= 0 and frame < ends[i]

    return inside
