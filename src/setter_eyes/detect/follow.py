"""Follow the ball from frame to frame, so tiled detection runs only where it is.

A full tiled scan costs one model pass per tile (9 for 3x3). While a moving ball is
followed, a frame needs only one tile-sized window centred on where the ball should be
(last position + velocity), so the model sees it at the same scale as in a tile. The
whole frame is still scanned
  - when nothing moving is being followed (before the serve, after a ball is lost),
  - after a new ball is found, until it has been seen `confirm_hits` times (so a serve
    seen in a periodic scan gets followed, not left to the next scan),
  - every `full_scan_every_s` anyway (a new ball: the serve from the far side),
  - on any frame where a window misses its ball (occluded, a sharp touch): the same frame
    is scanned again in full, so a miss costs one extra pass, not a lost sample.
Spare balls on the floor and marks on the wall never move, so they are never followed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from setter_eyes.datasets.tiles import tile_boxes

Window = tuple[int, int, int, int]                       # x, y, w, h
Dets = list[tuple[list[float], float]]                   # [(x1, y1, x2, y2), score]
WindowFn = Callable[[np.ndarray, list[Window]], Dets]    # (BGR, windows) -> detections


@dataclass
class _Ball:
    pos: np.ndarray
    vel: np.ndarray
    t: float
    origin: np.ndarray
    score: float
    hits: int = 1
    moved: float = 0.0

    def predict(self, t: float) -> np.ndarray:
        return self.pos + self.vel * (t - self.t)


class BallFollower:
    """Callable (BGR, frame index) -> detections, for detect_video. `cfg` is
    detect.ball_follow with its pixel settings already scaled to the video."""

    def __init__(self, detect_windows: WindowFn, grid: tuple[int, int], overlap: float,
                 fps: float, cfg: dict):
        self.detect_windows, self.grid, self.overlap = detect_windows, grid, overlap
        self.fps, self.cfg = fps, cfg
        self.balls: list[_Ball] = []
        self.last_full = -np.inf
        self.prev_t: float | None = None
        self.frames = self.passes = self.full_scans = 0

    def __call__(self, bgr: np.ndarray, index: int) -> Dets:
        c = self.cfg
        t = index / self.fps
        h, w = bgr.shape[:2]
        tiles = tile_boxes(w, h, self.grid, self.overlap)
        self.frames += 1
        self.balls = [b for b in self.balls if t - b.t <= c["forget_s"]]
        targets = self._targets()
        young = any(b.hits < c["confirm_hits"] and b.t == self.prev_t for b in self.balls)
        dets = None
        if targets and not young and t - self.last_full < c["full_scan_every_s"]:
            windows = self._windows([b.predict(t) for b in targets], tiles[0][2:], w, h)
            dets = self._run(bgr, windows)
            matched = self._assign(dets, t)
            if not all(id(b) in matched for b in targets):
                dets = None                   # a window missed its ball: scan in full
        if dets is None:
            dets = self._run(bgr, tiles)
            self.full_scans += 1
            self.last_full = t
            matched = self._assign(dets, t)
        self._commit(dets, matched, t)
        self.prev_t = t
        return dets

    def summary(self) -> str:
        n = max(1, self.frames)
        return (f"{self.passes / n:.2f} model passes per frame "
                f"({self.full_scans / n:.0%} of frames scanned in full)")

    def _run(self, bgr: np.ndarray, windows: list[Window]) -> Dets:
        self.passes += len(windows)
        return self.detect_windows(bgr, windows)

    def _targets(self) -> list[_Ball]:
        """Moving balls seen on the previous processed frame, most confident first."""
        c = self.cfg
        live = [b for b in self.balls if b.hits >= c["confirm_hits"]
                and b.moved >= c["min_move_px"]
                and b.t == self.prev_t]
        return sorted(live, key=lambda b: -b.score)[:c["max_windows"]]

    def _windows(self, points: list[np.ndarray], size: tuple[int, int], width: int,
                 height: int) -> list[Window]:
        tw, th = size
        m = self.cfg["edge_margin_px"]
        out: list[Window] = []
        for p in points:
            if any(x + m <= p[0] <= x + ww - m and y + m <= p[1] <= y + hh - m
                   for x, y, ww, hh in out):
                continue                      # already inside another window
            x = int(np.clip(round(p[0] - tw / 2), 0, width - tw))
            y = int(np.clip(round(p[1] - th / 2), 0, height - th))
            out.append((x, y, tw, th))
        return out

    def _assign(self, dets: Dets, t: float) -> dict[int, int]:
        """Greedy nearest match of detections to predicted balls -> {id(ball): det index}."""
        gate = self.cfg["gate_px"]
        pairs = []
        for b in self.balls:
            p = b.predict(t)
            for i, (box, _) in enumerate(dets):
                d = float(np.linalg.norm(_centre(box) - p))
                if d <= gate:
                    pairs.append((d, id(b), i))
        matched: dict[int, int] = {}
        used: set[int] = set()
        for _, bid, i in sorted(pairs):
            if bid not in matched and i not in used:
                matched[bid] = i
                used.add(i)
        return matched

    def _commit(self, dets: Dets, matched: dict[int, int], t: float) -> None:
        for b in self.balls:
            if id(b) not in matched:
                continue
            box, score = dets[matched[id(b)]]
            pos = _centre(box)
            vel = (pos - b.pos) / max(1e-6, t - b.t)
            b.vel = vel if b.hits == 1 else 0.5 * (b.vel + vel)
            b.pos, b.t, b.hits = pos, t, b.hits + 1
            b.score = max(b.score, score)
            b.moved = max(b.moved, float(np.linalg.norm(pos - b.origin)))
        taken = set(matched.values())
        for i, (box, score) in enumerate(dets):
            if i not in taken and score >= self.cfg["min_score"]:
                pos = _centre(box)
                self.balls.append(_Ball(pos, np.zeros(2), t, pos.copy(), score))


def _centre(box: list[float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
