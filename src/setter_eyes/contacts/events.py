"""Ball contacts from soft evidence (Milestone 4). No audio.

Candidates are the boundaries between consecutive ball flights (the ball changed course).
Each is scored by fusing, never by a single detector:
  - how sharply the ball changed course: velocity change between the two flights (m/s,
    from the 3D arcs; an image-space estimate when the arcs are weak)
  - whether someone could have touched it: horizontal distance from the ball to the nearest
    person's feet position, with the ball within reach height (or, without 3D, how close
    the ball is to a person's box in the image, in body heights)
  - floor: the ball low (< `floor_height_m`) after falling -> a floor contact, not a touch
Pose (wrist/forearm proximity) joins in Milestone 6; rally-state priors (<= 3 touches a
side) are applied per rally in rally/segment.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath
from setter_eyes.track.people import PeopleTracks


@dataclass
class Contact:
    path_index: int
    frame: int
    t: float
    kind: str                       # touch | floor
    team: str | None                # us | them | None
    track_id: int                   # toucher's people track, -1 if none
    ball_xyz: list[float] | None    # court metres, when a confident arc gives it
    ball_uv: list[float]            # image pixels
    velocity_change: float          # m/s (approximate when from the image)
    person_distance: float          # m (3D) or body heights (image), inf if nobody near
    confidence: float
    serve: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["person_distance"] = None if not np.isfinite(self.person_distance) else \
            round(self.person_distance, 2)
        return d


def _people_at(people: PeopleTracks, frames_index: dict, frame: int) -> np.ndarray:
    return frames_index.get(int(frame), np.zeros(0, int))


def find_contacts(path: BallPath, arcs: list[Arc], g: np.ndarray, fps: float,
                  people: PeopleTracks, roles: dict[int, dict], cfg: dict) -> list[Contact]:
    c = cfg["contacts"]
    order = np.argsort(people.frame, kind="stable")
    uniq, starts = np.unique(people.frame[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    rows_at = {int(f): order[a:b] for f, a, b in zip(uniq, starts, ends, strict=True)}
    times = path.frames / fps
    out: list[Contact] = []
    for prev, nxt in zip(arcs[:-1], arcs[1:], strict=True):
        if nxt.first - prev.last > c["max_join_gap_samples"]:
            continue                                      # ball lost between: not one event
        k = prev.last
        f = int(path.frames[k])
        t = float(times[k])
        uv = path.uv[k]
        strong = prev.confidence >= c["min_arc_confidence"] and \
            nxt.confidence >= c["min_arc_confidence"]
        xyz = None
        if strong:
            p_end = prev.position(np.array([t]), g)[0]
            p_start = nxt.position(np.array([t]), g)[0]
            xyz = (p_end + p_start) / 2
            v_end = np.array(prev.v0) + g * (t - prev.t0)
            v_start = np.array(nxt.v0) + g * (t - nxt.t0)
            dv = float(np.linalg.norm(v_start - v_end))
            falling = v_end[2] < 0
        else:
            dv = _image_velocity_change(path, k, fps, c)
            falling = False
            # One confident side still places the ball in 3D.
            for arc, before in ((prev, True), (nxt, False)):
                if arc.confidence >= c["min_arc_confidence"]:
                    xyz = arc.position(np.array([t]), g)[0]
                    if before:
                        falling = (np.array(arc.v0) + g * (t - arc.t0))[2] < 0
                    break
        rows = _people_at(people, rows_at, f)
        tid, dist = _nearest_person(people, rows, xyz, uv, c)
        p_break = 1 - np.exp(-dv / c["velocity_scale_mps"])
        if xyz is not None and xyz[2] <= c["floor_height_m"] and falling:
            kind, conf, team = "floor", p_break, _side(xyz)
        else:
            scale = c["reach_scale_m"] if xyz is not None else c["reach_scale_bodies"]
            p_person = float(np.exp(-dist / scale)) if np.isfinite(dist) else c["no_person_prior"]
            kind, conf = "touch", float(p_break * p_person)
            role = roles.get(tid, {}).get("role") if tid >= 0 else None
            team = "us" if role == "near_player" else "them" if role == "far_player" else \
                (_side(xyz) if xyz is not None else None)
        if conf < c["min_confidence"]:
            continue
        out.append(Contact(int(k), f, t, kind, team, int(tid),
                           None if xyz is None else [round(float(v), 3) for v in xyz],
                           [round(float(v), 1) for v in uv], round(dv, 2), dist,
                           round(float(conf), 3)))
    return out


def _side(xyz) -> str:
    return "us" if xyz[1] >= 0 else "them"


def _nearest_person(people: PeopleTracks, rows: np.ndarray, xyz, uv, c) -> tuple[int, float]:
    """-> (track id, distance): metres horizontally (ball within reach height) when 3D is
    known, else distance from the ball to the box in body heights."""
    best, best_d = -1, float("inf")
    for r in rows:
        if xyz is not None:
            if xyz[2] > c["max_reach_m"]:
                continue
            d = float(np.linalg.norm(people.court_xy_smooth[r] - xyz[:2]))
        else:
            x1, y1, x2, y2 = people.box[r]
            h = max(1.0, y2 - y1)
            # Allow overhead contacts: the reach zone extends above the head.
            top = y1 - c["overhead_reach_bodies"] * h
            dx = max(x1 - uv[0], 0.0, uv[0] - x2)
            dy = max(top - uv[1], 0.0, uv[1] - y2)
            d = float(np.hypot(dx, dy) / h)
        if d < best_d:
            best, best_d = int(people.track_id[r]), d
    return best, best_d


def _image_velocity_change(path: BallPath, k: int, fps: float, c: dict) -> float:
    """Rough m/s from the change of pixel velocity either side of sample k."""
    lo, hi = max(0, k - 3), min(len(path.frames) - 1, k + 3)
    if k - lo < 1 or hi - k < 1:
        return 0.0
    t = path.frames / fps
    v1 = (path.uv[k] - path.uv[lo]) / max(1e-6, t[k] - t[lo])
    v2 = (path.uv[hi] - path.uv[k]) / max(1e-6, t[hi] - t[k])
    dv_px = float(np.linalg.norm(v2 - v1))
    return 0.0 if not np.isfinite(dv_px) else dv_px * c["metres_per_px_at_typical_depth"]
