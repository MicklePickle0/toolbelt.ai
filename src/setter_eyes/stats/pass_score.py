"""Serve-receive pass quality 0-4 from where the pass is played (Milestone 8, first version).

Where the pass goes is taken from the player who plays it next (normally the setter):
their feet on the floor, at the moment of that touch (`pass_spot`). That player is found
  - by the ball's 3D position when a confident flight gives it: the player on our side
    whose feet are nearest the point under the ball;
  - else in the image: the player whose hands are at the ball (the ball level with the
    top of their box and inside it sideways), not merely the nearest box. A back-row
    player close to the camera has a big box that "reaches" a ball the setter is playing
    at the net, and can hide the setter (owner's example: the libero in front of the
    setter): picking by nearest box turned 6 of 8 big mismatches into wrong players.
The pass's spot is the feet, not the ball in the air: from behind the end line, the
ball's 3D depth is the weak axis (several perfect passes came out 8 m from the net),
while the setter's position follows the owner's scores closely (labelled: median
distance from the target 0.9 m for a 4, 1.9 m for a 3, 3.1 m for a 2, 4.6 m for a 1).

The score follows config/pass_scale.yaml: distance from `target` laterally and in depth,
graded by the first matching entry of `grades`. A pass the other side plays next went
straight over: `overpass_score` (0). Aces come from the point outcome, not here.
The grade boundaries are still the draft; scripts/eval_passes.py compares them with the
owner's labelled pass scores and suggests values.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from setter_eyes.ball.tracker import BallPath
from setter_eyes.track.people import PeopleTracks

OUR_SIDE_Y = 0.2          # feet at least this far on our side of the net (court metres)


def player_position(people: PeopleTracks, track_id: int, frame: int,
                    window: int = 4) -> np.ndarray | None:
    """Court (x, y) of a tracked person's feet at about `frame`."""
    sel = np.flatnonzero((people.track_id == track_id) & (np.abs(people.frame - frame) <= window))
    if not len(sel):
        return None
    xy = people.court_xy_smooth[sel[np.argmin(np.abs(people.frame[sel] - frame))]]
    return xy if np.isfinite(xy).all() else None


def _h(box: np.ndarray) -> float:
    return max(1.0, float(box[3] - box[1]))


def pass_spot(path: BallPath, people: PeopleTracks, frame: int,
              xyz: np.ndarray | None = None, max_reach_m: float = 2.5,
              hands_bodies: float = 0.25) -> np.ndarray | None:
    """Court (x, y) of the feet of the player who plays the ball at `frame`, or None."""
    rows = [r for r in np.flatnonzero(np.abs(people.frame - frame) <= 2)
            if np.isfinite(people.court_xy_smooth[r]).all()
            and people.court_xy_smooth[r][1] >= OUR_SIDE_Y]
    if not rows:
        return None
    k = int(np.argmin(np.abs(path.frames - frame)))
    u, v = path.uv[k]
    if xyz is not None and np.isfinite(xyz).all():
        # Only players plausibly under the ball in the image too: a flight fitted badly at
        # this moment must not pick someone on the other side of the picture.
        if np.isfinite([u, v]).all():
            seen = [r for r in rows if people.box[r][0] - 0.5 * _h(people.box[r]) <= u
                    <= people.box[r][2] + 0.5 * _h(people.box[r])]
        else:
            seen = rows
        d = [float(np.linalg.norm(people.court_xy_smooth[r] - xyz[:2])) for r in seen]
        if d and min(d) <= max_reach_m:
            return people.court_xy_smooth[seen[int(np.argmin(d))]]
    if not np.isfinite([u, v]).all():
        return None
    best, best_gap = None, np.inf
    for r in rows:
        x1, y1, x2, y2 = people.box[r]
        h = max(1.0, y2 - y1)
        if not x1 - hands_bodies * h <= u <= x2 + hands_bodies * h:
            continue                                   # not under the ball sideways
        gap = abs(v - y1) / h                          # ball level with the top of the box
        if gap < best_gap:
            best, best_gap = people.court_xy_smooth[r], gap
    return best if best_gap <= hands_bodies * 2 else None


def ball_xyz(arcs: list, g: np.ndarray, t: float, min_conf: float = 0.4,
             slack_s: float = 0.1) -> np.ndarray | None:
    """The ball's court position at t from a confident flight around it, or None."""
    for a in arcs:
        if a.confidence >= min_conf and a.t0 - slack_s <= t <= a.t0 + a.duration_s + slack_s:
            return a.position(np.array([min(max(t, a.t0), a.t0 + a.duration_s)]), g)[0]
    return None


def score_passes(rallies: list, contacts: list, path: BallPath, people: PeopleTracks,
                 arcs: list, g: np.ndarray, fps: float, scale: dict) -> Counter:
    """Pass score and confidence on each rally's serve receive of ours (the first touch
    of ours after their serve, `receive` or `overpass`), in place on the Contact:
      - our next touch: graded from where it is played (`pass_spot`, `grade`);
      - their next touch: the pass went straight over, `overpass_score`;
      - no next touch and they won the point: shanked, `ace_score`;
      - no next touch otherwise: not scored.
    Confidence: how the spot was found (`pipeline:` in pass_scale.yaml) times how sure the
    receive's action and the next touch are. Returns counts by how each was scored."""
    p = scale["pipeline"]
    how = Counter()
    for c in contacts:                             # touches read back from an earlier run
        c.pass_score = c.pass_confidence = None
    for r in rallies:
        if r.serving_team != "them":
            continue
        touches = [contacts[i] for i in r.contacts if contacts[i].kind == "touch"]
        k = next((j for j, c in enumerate(touches) if c.team == "us"
                  and c.action in ("receive", "overpass")), None)
        if k is None:
            continue
        rec = touches[k]
        sure = rec.action_confidence if rec.action_confidence is not None else rec.confidence
        nxt = touches[k + 1] if k + 1 < len(touches) else None
        if nxt is None:
            if r.point_winner == "them":
                rec.pass_score, rec.pass_confidence = scale["ace_score"], \
                    round(p["shank_confidence"] * sure, 3)
                how["shank"] += 1
            else:
                how["unscored"] += 1
            continue
        if nxt.team != "us":
            rec.pass_score, rec.pass_confidence = scale["overpass_score"], \
                round(p["overpass_confidence"] * sure, 3)
            how["overpass"] += 1
            continue
        xyz = ball_xyz(arcs, g, nxt.t)
        xy = pass_spot(path, people, nxt.frame, xyz)
        if xy is None:
            how["no player found"] += 1
            continue
        by_3d = xyz is not None and np.isfinite(xyz).all()
        conf = p["located_3d_confidence" if by_3d else "located_image_confidence"] * \
            min(sure, nxt.confidence)
        if nxt.inferred:
            conf *= p["inferred_next_factor"]
        rec.pass_score, rec.pass_confidence = grade(xy, scale), round(conf, 3)
        how["graded"] += 1
    return how


def grade(xy: np.ndarray, scale: dict) -> int:
    """0-4 from where the pass was played (court metres)."""
    lat = abs(xy[0] - scale["target"]["x"])
    dep = abs(xy[1] - scale["target"]["y"])
    for g in scale["grades"]:
        if (g["max_depth_error"] is None or dep <= g["max_depth_error"]) and \
                (g["max_lateral_error"] is None or lat <= g["max_lateral_error"]):
            return int(g["score"])
    return 1
