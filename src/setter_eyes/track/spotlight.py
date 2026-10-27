"""Spotlight one player in a clip: find them, follow them, draw a ring around their feet.

For a player's highlights (scripts/clips.py --player): jersey numbers aren't read yet
(Milestone 7), but the labels say when that player touched the ball, so at that moment
they are the near-side person nearest the ball (`toucher_track`): by court distance to
the point under the ball when a confident flight gives its 3D position, else in the
image. (Image distance alone picked a back-row player near the camera for attacks at the
net: a big box near the lens "reaches" a ball far away.) Their track is then
followed through the clip; when it breaks, the track that starts closest to where it
ended, soon after, carries on (`follow`).

The ring is Hudl style: a solid band on the floor around the feet (`ring_radius_m`,
`ring_width_m` in court metres, projected through the camera so it lies on the court),
not a filled disc. Its back part, behind the player, is hidden inside their box above the
feet, so the legs stand in front of it and it reads as a ring around them.
"""

from __future__ import annotations

import cv2
import numpy as np

from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath
from setter_eyes.calibration.camera import Camera
from setter_eyes.track.people import PeopleTracks

# Feet at least this far on our side of the net (court metres): an opponent blocking just
# across the net must not be taken for our attacker (track.roles uses the same clearance).
OUR_SIDE_Y = 0.2


def ball_xyz(arcs: list[Arc], g: np.ndarray, t: float, min_conf: float = 0.4,
             slack_s: float = 0.1) -> np.ndarray | None:
    """The ball's court position at t from a confident flight around it, or None."""
    for a in arcs:
        if a.confidence >= min_conf and a.t0 - slack_s <= t <= a.t0 + a.duration_s + slack_s:
            tt = min(max(t, a.t0), a.t0 + a.duration_s)
            return a.position(np.array([tt]), g)[0]
    return None


def toucher_track(path: BallPath, people: PeopleTracks, frame: int,
                  max_bodies: float = 0.6, xyz: np.ndarray | None = None,
                  max_reach_m: float = 2.0) -> int | None:
    """Track id of the near-side person nearest the ball at `frame`, or None when nobody
    is close enough. With the ball's 3D position (`xyz`): the person whose feet are
    nearest the point under it (within `max_reach_m`); else by image distance (reach above
    the head included)."""
    if xyz is not None and np.isfinite(xyz).all():
        rows = np.flatnonzero(np.abs(people.frame - frame) <= 2)
        best, best_d = None, np.inf
        for r in rows:
            xy = people.court_xy_smooth[r]
            if not np.isfinite(xy).all() or xy[1] < OUR_SIDE_Y:              # our side only
                continue
            d = float(np.linalg.norm(xy - xyz[:2]))
            if d < best_d:
                best, best_d = int(people.track_id[r]), d
        if best is not None and best_d <= max_reach_m:
            return best
    k = int(np.argmin(np.abs(path.frames - frame)))
    uv = path.uv[k]
    rows = np.flatnonzero(np.abs(people.frame - frame) <= 2)
    if not np.isfinite(uv).all() or not len(rows):
        return None
    best, best_d = None, np.inf
    for r in rows:
        x1, y1, x2, y2 = people.box[r]
        h = max(1.0, y2 - y1)
        d = np.hypot(max(x1 - uv[0], 0.0, uv[0] - x2),
                     max(y1 - 0.8 * h - uv[1], 0.0, uv[1] - y2)) / h
        xy = people.court_xy_smooth[r]
        if d < best_d and np.isfinite(xy).all() and xy[1] >= OUR_SIDE_Y:  # our side
            best, best_d = int(people.track_id[r]), d
    return best if best_d <= max_bodies else None


def follow(people: PeopleTracks, track_id: int, f0: int, f1: int, fps: float,
           max_gap_s: float = 1.0, max_speed_mps: float = 6.0, slack_m: float = 0.7
           ) -> list[int]:
    """Rows of `people` for the player over frames [f0, f1]: their track, continued by the
    nearest plausible track whenever it ends (both directions), in frame order."""
    by_track: dict[int, np.ndarray] = {}
    for tid in np.unique(people.track_id):
        rows = np.flatnonzero(people.track_id == tid)
        by_track[int(tid)] = rows[np.argsort(people.frame[rows])]
    used = {track_id}
    chain = list(by_track[track_id])

    def extend(forward: bool) -> None:
        while True:
            end = chain[-1] if forward else chain[0]
            fe, xy = people.frame[end], people.court_xy_smooth[end]
            if (forward and fe >= f1) or (not forward and fe <= f0):
                return
            best, best_d = None, np.inf
            for tid, rows in by_track.items():
                if tid in used:
                    continue
                r = rows[0] if forward else rows[-1]
                gap = (people.frame[r] - fe) / fps * (1 if forward else -1)
                if not 0 < gap <= max_gap_s:
                    continue
                d = float(np.linalg.norm(people.court_xy_smooth[r] - xy))
                if d <= slack_m + max_speed_mps * gap and d < best_d:
                    best, best_d = tid, d
            if best is None:
                return
            used.add(best)
            if forward:
                chain.extend(by_track[best])
            else:
                chain[:0] = list(by_track[best])

    extend(True)
    extend(False)
    return [r for r in chain if f0 - 4 <= people.frame[r] <= f1 + 4]


def position_at(people: PeopleTracks, rows: list[int], frame: int
                ) -> tuple[np.ndarray, np.ndarray] | None:
    """(court xy, box) of the followed player at `frame`, interpolated between samples."""
    if not rows:
        return None
    f = people.frame[rows]
    if frame < f[0] - 4 or frame > f[-1] + 4:
        return None
    j = int(np.clip(np.searchsorted(f, frame), 1, len(rows) - 1)) if len(rows) > 1 else 0
    a, b = rows[max(0, j - 1)], rows[j]
    fa, fb = people.frame[a], people.frame[b]
    w = 0.0 if fb == fa else float(np.clip((frame - fa) / (fb - fa), 0, 1))
    xy = (1 - w) * people.court_xy_smooth[a] + w * people.court_xy_smooth[b]
    box = (1 - w) * people.box[a] + w * people.box[b]
    return (xy, box) if np.isfinite(xy).all() else None


def draw_ring(image: np.ndarray, cam: Camera, xy: np.ndarray, box: np.ndarray | None,
              cfg: dict, s: float = 1.0) -> None:
    """A band on the floor around the feet at court `xy`. cam is the full-resolution camera,
    s scales its pixels to `image`; `box` (full resolution) hides the back of the ring
    behind the player's legs."""
    r_out = cfg["ring_radius_m"]
    r_in = r_out - cfg["ring_width_m"]
    a = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rings = []
    for r in (r_out, r_in):
        pts = np.stack([xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), np.zeros_like(a)], axis=1)
        uv, z = cam.project(pts)
        if (z <= 0.1).any():
            return
        rings.append((uv * s).astype(np.float32))
    outer, inner = rings
    h, w = image.shape[:2]
    band = np.zeros((h, w), np.uint8)
    cv2.fillPoly(band, [outer.round().astype(np.int32)], 255, cv2.LINE_AA)
    cv2.fillPoly(band, [inner.round().astype(np.int32)], 0, cv2.LINE_AA)
    if box is not None:
        # Hide the back of the ring where the player stands in front of it: inside their
        # box, above their feet (the front of the ring is below the feet).
        feet_v = int(round(float(cam.project(np.array([[xy[0], xy[1], 0.0]]))[0][0, 1] * s)))
        x1, y1, x2, _ = (box * s).round().astype(int)
        band[max(0, y1):max(0, feet_v), max(0, x1):max(0, x2)] = 0
    alpha = (band.astype(np.float32) / 255.0 * cfg["ring_opacity"])[..., None]
    colour = np.array(cfg["ring_colour"], np.float32)
    image[:] = (image * (1 - alpha) + colour * alpha).astype(np.uint8)
