"""Run a detector over every Nth frame of a whole video, resumably.

Writes an .npz with
  frames      every processed frame index (including frames with no detection); with
              `keep` (e.g. play spans only) the skipped frames are missing, not empty
  det_frame   frame index of each detection, det_box (x1, y1, x2, y2) px, det_score
plus `meta` (stride, threshold, settings, ...). Saved every `save_every` frames: a rerun
with the same stride, threshold and settings continues after the last saved frame unless
`restart`. Frames are decoded on a background thread so the GPU never waits for the next one.

With a `planner` (detect/roi.py, detect_ball.py --roi) each frame is searched as planned:
everywhere (`detect`), in one crop around the ball (`detect_crop`), or skipped while idle;
skipped frames are kept in memory for `backfill_s` and searched when the ball turns up
again. Frames the planner skips still count as processed (with no detection), as in the
simulation (scripts/simulate_roi.py).
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from setter_eyes.ingest import iter_frames

DetectFn = Callable[[np.ndarray, int], list[tuple[list[float], float]]]
# (BGR, frame index) -> [(box, score)]
CropFn = Callable[[np.ndarray, tuple[int, int, int, int]], list[tuple[list[float], float]]]
# (BGR, crop x, y, w, h) -> [(box, score)] in full-image pixels


def _save(path: Path, frames, det_frame, det_box, det_score, meta: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, frames=np.array(frames, np.int64),
                        det_frame=np.array(det_frame, np.int64),
                        det_box=np.array(det_box, np.float32).reshape(-1, 4),
                        det_score=np.array(det_score, np.float32), **meta)
    tmp.replace(path)                     # never leave a half-written file behind


def detect_video(video: Path, out: Path, detect: DetectFn, *, stride: int, threshold: float,
                 fps: float, total_frames: int, first: int = 0, last: int | None = None,
                 save_every: int = 2000, restart: bool = False, meta: dict | None = None,
                 keep: Callable[[int], bool] | None = None, settings: str = "",
                 log: Callable[[str], None] = print, planner=None,
                 detect_crop: CropFn | None = None) -> int:
    """-> number of frames processed in this run. `keep(frame)` limits which frames on
    the stride grid are detected; `settings` names anything else that changes the
    output (a rerun with different settings starts over); `planner` + `detect_crop`
    search near the ball (see the module docstring)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    frames, det_frame, det_box, det_score = [], [], [], []
    if out.exists() and not restart:
        with np.load(out) as prev:        # closed again: Windows can't replace an open file
            # A run from before settings were stored detected every frame in full: a
            # superset of any settings, so it is continued, not thrown away.
            prev_settings = str(prev["settings"]) if "settings" in prev else settings
            same = int(prev["stride"]) == stride and float(prev["threshold"]) == threshold \
                and prev_settings == settings
            if same:
                frames, det_frame = list(prev["frames"]), list(prev["det_frame"])
                det_box, det_score = list(prev["det_box"]), list(prev["det_score"])
        if same:
            if frames:
                first = max(first, int(frames[-1]) + 1)
                log(f"resuming after frame {frames[-1]} ({len(frames)} frames done)")
        else:
            log("previous run used other settings: starting over")
    first += (-first) % stride            # keep the stride grid aligned when resuming
    meta = {**(meta or {}), "stride": stride, "threshold": threshold, "fps": fps,
            "settings": settings}

    q: queue.Queue = queue.Queue(maxsize=8)

    def reader():
        for frame in iter_frames(video, start=first, end=last, step=stride, keep=keep):
            q.put(frame)
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    t0, n = time.time(), 0
    todo = max(1, sum(1 for f in range(first, last or total_frames, stride)
                      if keep is None or keep(f)))
    runs = {"full": 0, "crop": 0, "skip": 0}
    skipped: list = []                    # frames kept for backfill while idle

    def add(index: int, found) -> None:
        for box, score in found:
            det_frame.append(index)
            det_box.append(box)
            det_score.append(score)

    while (frame := q.get()) is not None:
        t = frame.index / fps
        if planner is None:
            add(frame.index, detect(frame.image, frame.index))
        else:
            was_idle = planner.idle(t)
            mode, crop = planner.plan(t)
            runs[mode] += 1
            if mode == "skip":
                skipped.append(frame)
            else:
                found = detect(frame.image, frame.index) if mode == "full" else \
                    detect_crop(frame.image, crop)
                add(frame.index, found)
                planner.update(t, np.array([b for b, _ in found]).reshape(-1, 4),
                               np.array([s for _, s in found]))
                if was_idle and not planner.idle(t):      # back: search what was skipped
                    backfilled = False
                    for old in skipped:
                        if t - old.index / fps <= planner.cfg["backfill_s"]:
                            add(old.index, detect(old.image, old.index))
                            backfilled = True
                    skipped = []
                    if backfilled:                    # those came late: back in order
                        order = np.argsort(det_frame, kind="stable")
                        det_frame = [det_frame[i] for i in order]
                        det_box = [det_box[i] for i in order]
                        det_score = [det_score[i] for i in order]
            skipped = [s for s in skipped if t - s.index / fps <= planner.cfg["backfill_s"]]
        frames.append(frame.index)
        n += 1
        if n % save_every == 0:
            _save(out, frames, det_frame, det_box, det_score, meta)
            rate = n / (time.time() - t0)
            log(f"  frame {frame.index}: {n}/{todo} at {rate:.1f}/s, "
                f"~{(todo - n) / rate / 60:.0f} min left")
    _save(out, frames, det_frame, det_box, det_score, meta)
    log(f"done: {len(frames)} frames, {len(det_frame)} detections "
        f"({n / max(1e-9, time.time() - t0):.1f} frames/s) -> {out}")
    if planner is not None:
        log("near-the-ball search, frames: " + ", ".join(
            f"{k} {v / max(1, n):.0%}" for k, v in runs.items()))
    return n
