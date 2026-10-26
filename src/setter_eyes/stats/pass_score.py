"""Serve-receive pass quality 0-4 from where the pass is played (Milestone 8, first version).

Where the pass goes is taken from the player who plays it next (normally the setter):
their feet on the floor, at the moment of that touch. Feet on the floor are much better
placed than the ball in the air: from behind the end line, the ball's 3D depth is the
weak axis (several perfect passes came out 8 m from the net), while the setter's
position follows the owner's scores closely (labelled: median distance from the target
0.9 m for a 4, 1.9 m for a 3, 3.1 m for a 2, 4.6 m for a 1).

The score follows config/pass_scale.yaml: distance from `target` laterally and in depth,
graded by the first matching entry of `grades`. A pass the other side plays next went
straight over: `overpass_score` (0). Aces come from the point outcome, not here.
The grade boundaries are still the draft; scripts/eval_passes.py compares them with the
owner's labelled pass scores and suggests values.
"""

from __future__ import annotations

import numpy as np

from setter_eyes.track.people import PeopleTracks


def player_position(people: PeopleTracks, track_id: int, frame: int,
                    window: int = 4) -> np.ndarray | None:
    """Court (x, y) of a tracked person's feet at about `frame`."""
    sel = np.flatnonzero((people.track_id == track_id) & (np.abs(people.frame - frame) <= window))
    if not len(sel):
        return None
    xy = people.court_xy_smooth[sel[np.argmin(np.abs(people.frame[sel] - frame))]]
    return xy if np.isfinite(xy).all() else None


def grade(xy: np.ndarray, scale: dict) -> int:
    """0-4 from where the pass was played (court metres)."""
    lat = abs(xy[0] - scale["target"]["x"])
    dep = abs(xy[1] - scale["target"]["y"])
    for g in scale["grades"]:
        if (g["max_depth_error"] is None or dep <= g["max_depth_error"]) and \
                (g["max_lateral_error"] is None or lat <= g["max_lateral_error"]):
            return int(g["score"])
    return 1
