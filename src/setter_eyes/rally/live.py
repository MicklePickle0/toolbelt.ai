"""Is play still on? The live-ball check on a rally's trailing touches.

After the whistle the ball is often caught, held or tossed back, and those look like
touches; a net touch or a lift is only a whistle (no audio here). A trailing touch is
taken as dead, and the rally ended before it, when BOTH:
  - it comes more than `max_gap_s` after the previous touch (labelled rallies: touches
    are at most 2.2 s apart), and
  - the near-side players have slowed: their mean speed in the `window_s` before it is
    below `motion_ratio` x their mean speed over the rally so far.
On both labelled sets this caught 5 of 13 dead touches and cut none of 76 real last
touches; looser rules started cutting real kills, which costs more than a stray touch.
Not used yet: the ball leaving slowly (caught), and the net shaking with the ball away
from it (a net touch) - both in docs/todo.md.
"""

from __future__ import annotations

import numpy as np

from setter_eyes.track.people import PeopleTracks


class PlayerMotion:
    """Mean speed of the people standing in our half, over any time window."""

    def __init__(self, people: PeopleTracks, fps: float, half_margin_m: float = 1.0,
                 max_dt_s: float = 0.2):
        xy = people.court_xy_smooth
        m = half_margin_m
        ours = (xy[:, 0] > -m) & (xy[:, 0] < 9 + m) & (xy[:, 1] > 0) & (xy[:, 1] < 9 + m)
        order = np.lexsort((people.frame, people.track_id))
        f, tid, p, ours = people.frame[order], people.track_id[order], xy[order], ours[order]
        dt = np.diff(f) / fps
        step = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(dt, 1e-6)
        ok = (np.diff(tid) == 0) & (dt > 0) & (dt < max_dt_s) & ours[1:]
        t = f[1:][ok] / fps
        sort = np.argsort(t)
        self.t, self.speed = t[sort], step[ok][sort]

    def mean(self, t0: float, t1: float) -> float:
        a, b = np.searchsorted(self.t, [t0, t1])
        v = self.speed[a:b]
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")


def dead_from(touch_t: list[float], start_t: float, motion: PlayerMotion | None,
              cfg: dict) -> int:
    """Index of the first dead trailing touch (len(touch_t) if all live)."""
    if motion is None:
        return len(touch_t)
    cut = len(touch_t)
    for i in range(len(touch_t) - 1, 0, -1):          # from the end, never the first
        t = touch_t[i]
        if t - touch_t[i - 1] <= cfg["max_gap_s"]:
            break
        base = motion.mean(start_t, t - cfg["window_s"])
        now = motion.mean(t - cfg["window_s"], t)
        if not (np.isfinite(base) and np.isfinite(now) and base > 0
                and now < cfg["motion_ratio"] * base):
            break
        cut = i
    return cut
