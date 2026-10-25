"""Evidence per touch candidate, for the learned scorer (contacts/scorer.py).

A candidate is a break between two ball flights (contacts/events.py). Hand-made fusion of
course change and person proximity barely separated real touches from the rest on
labelled footage (most false alarms are a second break around the same touch, or a person
standing near a passing ball), so the scorer learns from these instead:
  - the course change: 3D velocity change; image speed in and out, their ratio, the turn
    (cosine), image vertical velocity in and out, falling-then-rising
  - where the ball is: court x, y, z and |y| (when a flight gives 3D), apparent size in
    pixels (the strongest depth cue: far-side balls are small)
  - who is near: 3D distance to the nearest person's feet; image distance to the nearest
    box in body heights (reach zone above the head included), where on that body the ball
    is (<0 above the head), how many people are within 1.5 bodies, that person's court y
  - timing: time to the previous and next candidate (a touch has one break; a split flight
    has two close together)
  - the heuristic's own team call (1 us, 0 them, 0.5 unknown), for the team model
"""

from __future__ import annotations

import numpy as np

from setter_eyes.ball.tracker import BallPath
from setter_eyes.contacts.events import Contact
from setter_eyes.track.people import PeopleTracks

NAMES = ["dv", "person_dist_m", "x", "y", "z", "abs_y", "speed_in", "speed_out", "speed_ratio",
         "turn_cos", "vy_in", "vy_out", "fall_then_rise", "person_dist_bodies", "on_body",
         "people_close", "person_y", "prev_gap_s", "next_gap_s", "size_px", "team_heuristic"]
_WINDOW = 4                       # samples either side for image velocities
_FAR = 20.0                       # cap for "nobody near"
_GAP_CAP = 9.0


def _rows_by_frame(people: PeopleTracks) -> dict[int, np.ndarray]:
    order = np.argsort(people.frame, kind="stable")
    uniq, starts = np.unique(people.frame[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    return {int(f): order[a:b] for f, a, b in zip(uniq, starts, ends, strict=True)}


def candidate_features(cands: list[Contact], path: BallPath, people: PeopleTracks,
                       fps: float, overhead_bodies: float = 0.6) -> np.ndarray:
    """(N, len(NAMES)) features; cands in time order."""
    t = path.frames / fps
    ct = np.array([c.t for c in cands])
    rows_at = _rows_by_frame(people)

    def vel(k0: int, k1: int) -> np.ndarray:
        if k0 < 0 or k1 >= len(t) or k1 <= k0:
            return np.full(2, np.nan)
        return (path.uv_ref[k1] - path.uv_ref[k0]) / (t[k1] - t[k0])

    out = np.full((len(cands), len(NAMES)), np.nan)
    for i, c in enumerate(cands):
        k = c.path_index
        vin, vout = vel(k - _WINDOW, k), vel(k, k + _WINDOW)
        s_in, s_out = np.linalg.norm(vin), np.linalg.norm(vout)
        best, on_body, close, person_y = _FAR, np.nan, 0, np.nan
        u, v = c.ball_uv
        for r in rows_at.get(int(c.frame), ()):
            x1, y1, x2, y2 = people.box[r]
            h = max(1.0, y2 - y1)
            dx = max(x1 - u, 0.0, u - x2)
            dy = max(y1 - overhead_bodies * h - v, 0.0, v - y2)
            d = float(np.hypot(dx, dy) / h)
            close += d < 1.5
            if d < best:
                best, on_body = d, (v - y1) / h
                person_y = float(people.court_xy_smooth[r][1])
        xyz = c.ball_xyz if c.ball_xyz is not None else [np.nan] * 3
        team = {"us": 1.0, "them": 0.0}.get(c.team, 0.5)
        out[i] = [c.velocity_change, min(c.person_distance, _FAR), *xyz, abs(xyz[1]),
                  s_in, s_out, s_out / (s_in + 1e-6), np.dot(vin, vout) / (s_in * s_out + 1e-9),
                  vin[1], vout[1], float(vin[1] > 0 and vout[1] < 0), best, on_body, close,
                  person_y, min(ct[i] - ct[i - 1], _GAP_CAP) if i else _GAP_CAP,
                  min(ct[i + 1] - ct[i], _GAP_CAP) if i + 1 < len(ct) else _GAP_CAP,
                  float(np.nan_to_num(path.size_px[k], nan=0.0)), team]
    return out
