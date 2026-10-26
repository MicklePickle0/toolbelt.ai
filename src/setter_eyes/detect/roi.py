"""Search for the ball near where it was, not over the whole frame (GPU speed-up, step 1).

Ball detection runs the detector on 9 tiles per frame (3x3 over 4K), ~2 h per set. Once
the ball is found it moves predictably, so the next frame only needs ONE crop, the size
of a tile, centred on its last confident position, as long as the distance it can have
travelled since (`max_speed_px_s` x time + `margin_px`) stays inside that crop. All 9
tiles are still searched
  - when no confident ball (score >= `min_score`) was seen for `lost_after_s`,
  - when it could have left the crop, and
  - every `full_every_s` regardless, to pick up a serve or the match ball after another
    ball held the search.
Idle mode: with no confident ball for `idle_after_s` (dead time between points), only
every `idle_every_s` is searched; when the ball turns up again, the frames skipped in the
`backfill_s` before it are searched after all, so a serve's first moments aren't lost.

The detector sees each tile or crop on its own, so a crop finds about what the full search
finds inside it: replaying saved full-search detections through this planner (simulate())
shows the effect without the GPU (scripts/simulate_roi.py). Not used by detect_ball.py yet.
"""

from __future__ import annotations

import numpy as np

from setter_eyes.ball.tracker import Detections
from setter_eyes.datasets.tiles import tile_boxes

FULL, CROP, SKIP = "full", "crop", "skip"


class RoiPlanner:
    def __init__(self, width: int, height: int, grid: tuple[int, int], overlap: float,
                 cfg: dict):
        self.width, self.height = width, height
        self.tiles = tile_boxes(width, height, grid, overlap)      # (x, y, w, h)
        self.cw, self.ch = self.tiles[0][2], self.tiles[0][3]     # a crop is one tile
        self.cfg = cfg
        self.last_xy: np.ndarray | None = None
        self.last_t = -np.inf
        self.last_full = -np.inf

    def idle(self, t: float) -> bool:
        return t - self.last_t > self.cfg["idle_after_s"]

    def plan(self, t: float) -> tuple[str, tuple[int, int, int, int] | None]:
        """(FULL, None) | (CROP, (x, y, w, h)) | (SKIP, None) for the frame at time t."""
        c = self.cfg
        if self.idle(t):
            if t - self.last_full < c["idle_every_s"]:
                return SKIP, None
            self.last_full = t
            return FULL, None
        reach = c["margin_px"] + c["max_speed_px_s"] * (t - self.last_t)
        if self.last_xy is None or t - self.last_t > c["lost_after_s"] or \
                t - self.last_full >= c["full_every_s"] or \
                reach > min(self.cw, self.ch) / 2:
            self.last_full = t
            return FULL, None
        x = int(np.clip(self.last_xy[0] - self.cw / 2, 0, self.width - self.cw))
        y = int(np.clip(self.last_xy[1] - self.ch / 2, 0, self.height - self.ch))
        return CROP, (x, y, self.cw, self.ch)

    def update(self, t: float, boxes: np.ndarray, scores: np.ndarray) -> None:
        """Detections found at time t (full-image boxes x1, y1, x2, y2)."""
        if len(scores) and scores.max() >= self.cfg["min_score"]:
            b = boxes[int(np.argmax(scores))]
            self.last_xy = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            self.last_t = t


def _inside(boxes: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return (boxes[:, 0] >= x) & (boxes[:, 1] >= y) & (boxes[:, 2] <= x + w) & \
        (boxes[:, 3] <= y + h)


def simulate(dets: Detections, width: int, height: int, grid: tuple[int, int],
             overlap: float, cfg: dict) -> tuple[Detections, float, dict]:
    """Full-search detections -> (what the near-the-ball search would have found, share of
    detector runs it needs vs the full search, share of frames per mode)."""
    planner = RoiPlanner(width, height, grid, overlap, cfg)
    n_tiles = len(planner.tiles)
    groups = dets.by_frame()
    keep: list[int] = []
    runs = 0
    modes = {FULL: 0, CROP: 0, SKIP: 0}
    skipped: list[float] = []                     # frames skipped while idle
    for f in dets.frames:
        t = f / dets.fps
        was_idle = planner.idle(t)
        mode, crop = planner.plan(t)
        modes[mode] += 1
        if mode == SKIP:
            skipped.append(f)
            continue
        runs += n_tiles if mode == FULL else 1
        idx = np.array(groups.get(int(f), []), int)
        if len(idx) and mode == CROP:
            idx = idx[_inside(dets.det_box[idx], *crop)]
        keep.extend(idx.tolist())
        planner.update(t, dets.det_box[idx], dets.det_score[idx])
        if was_idle and not planner.idle(t):      # back from idle: search what was skipped
            for g in skipped:
                if t - g / dets.fps <= cfg["backfill_s"]:
                    runs += n_tiles
                    keep.extend(groups.get(int(g), []))
            skipped = []
        elif was_idle:
            skipped = [g for g in skipped if t - g / dets.fps <= cfg["backfill_s"]]
    keep_arr = np.array(sorted(set(keep)), int)
    out = Detections(dets.frames, dets.det_frame[keep_arr], dets.det_box[keep_arr],
                     dets.det_score[keep_arr], dets.fps)
    total = len(dets.frames)
    return out, runs / (total * n_tiles), {k: v / total for k, v in modes.items()}
