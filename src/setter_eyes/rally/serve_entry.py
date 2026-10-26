"""Our serve, from how the ball enters the frame (the owner's idea, 2026-10-26).

Our server stands just behind the camera, so the serve itself is rarely seen: the hit is
off-screen or a blur. But right after it, the ball always shows up the same way: near
the TOP of the frame, very BIG (a metre or two from the lens; 100-200 px in 4K against
20-40 px in play), coming down over the server's head. An "our serve" moment is:
  1. the ball first seen big (>= `min_size` of the image height) in the top
     `top_share` of the frame,
  2. after `quiet_s` with no touch (play was dead: a serve starts from dead time), and
  3. far away soon after: smaller than `away_size` within `away_within_s` (it was hit
     away, not tossed back to the server, who also catches it overhead).
Labelled (titan-rise_set1 + mac-chrome_set1, 33 of our serves): today's touches found 22;
this finds 28, with 4 false alarms over both sets; timing ~0.3 s from the labelled hit.
Their serve isn't needed here: its toss and hit are in view (44 of 48 found).

The moment returned is the HIT, not the first sighting: the ball is often first seen
during the toss, rising up the frame at a steady size; the hit is where it stops rising
(then it shrinks away). Labelled: the first sighting was 0.4-0.8 s early on 10 of 29
serves; the stop-rising moment on none, and within 0.25 s of the hit on 62%. A ball
first seen after the hit (already coming down) can't be placed earlier.
Depends on the camera standing behind our end line (all Humber footage).
"""

from __future__ import annotations

import numpy as np

from setter_eyes.ball.tracker import BallPath


def _hit_time(path: BallPath, idx: np.ndarray, t: np.ndarray, k: int, image_height: int,
              cfg: dict) -> float:
    """Follow the ball from sighting k while it rises up the frame (the toss): the hit is
    where it stops rising."""
    follow = idx[(t[idx] >= t[k]) & (t[idx] <= t[k] + cfg["toss_max_s"])]
    slack = cfg["rise_slack"] * image_height
    j = 0
    while j + 1 < len(follow) and path.uv[follow[j + 1], 1] < path.uv[follow[j], 1] - slack:
        j += 1
    return float(t[follow[j]]) if len(follow) else float(t[k])


def find_our_serves(path: BallPath, fps: float, touch_t: np.ndarray, image_height: int,
                    cfg: dict) -> list[float]:
    """Times (s) the ball first appears after our serves."""
    t = path.frames / fps
    seen = np.isfinite(path.uv).all(axis=1) & path.measured
    idx = np.flatnonzero(seen)
    size = path.size_px / image_height
    big = idx[(size[idx] >= cfg["min_size"]) &
              (path.uv[idx, 1] < cfg["top_share"] * image_height)]
    touch_t = np.sort(np.asarray(touch_t, float))
    out: list[float] = []
    last = -np.inf
    for k in big:
        if t[k] - last <= cfg["merge_s"]:            # the same appearance, still big
            last = t[k]
            continue
        last = t[k]
        prev = touch_t[touch_t < t[k] - cfg["touch_slack_s"]]
        if len(prev) and t[k] - prev[-1] < cfg["quiet_s"]:
            continue                                  # play was on: not a serve
        after = idx[(t[idx] > t[k]) & (t[idx] <= t[k] + cfg["away_within_s"])]
        if not len(after) or not np.nanmin(size[after]) <= cfg["away_size"]:
            continue                                  # stayed close: caught, not served
        out.append(_hit_time(path, idx, t, k, image_height, cfg))
    return out
