"""Who touched the ball: the player whose hands or forearms are at it (owner's B2).

The nearest feet picked the wrong player for ~1/3 of labelled touches: a back-row player
close to the camera has a big box that "reaches" a ball played further away, and players
overlap in a low camera's view. Here each of our players' pose (scripts/pose_people.py,
10 Hz) is interpolated to the touch time, and the ball's image position is measured to
their forearms (elbow to wrist) and hands (the forearm carried on past the wrist by
`hand_extension`), in body heights (box height). The nearest within `reach_bodies` wins;
its confidence grows with the gap to the second nearest.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from setter_eyes.identity.numbers import L_ELBOW, L_WRIST, R_ELBOW, R_WRIST


@dataclass
class Poses:
    frame: np.ndarray        # (N,)
    track_id: np.ndarray     # (N,)
    box: np.ndarray          # (N, 4) x0 y0 x1 y1
    kp: np.ndarray           # (N, 17, 2) image pixels (COCO keypoints)
    kp_score: np.ndarray     # (N, 17)
    torso_lab: np.ndarray | None = None   # (N, 3) mean torso colour (libero check), NaN = none

    def save(self, path: Path) -> None:
        extra = {} if self.torso_lab is None else {"torso_lab": self.torso_lab.astype(np.float16)}
        np.savez_compressed(path, frame=self.frame, track_id=self.track_id, box=self.box,
                            kp=self.kp.astype(np.float32),
                            kp_score=self.kp_score.astype(np.float16), **extra)

    @classmethod
    def load(cls, path: Path) -> Poses:
        d = np.load(path)
        lab = d["torso_lab"].astype(np.float32) if "torso_lab" in d else None
        return cls(d["frame"], d["track_id"], d["box"], d["kp"].astype(np.float32),
                   d["kp_score"].astype(np.float32), lab)

    def colour_at(self, track_id: int, frame: int, max_df: int) -> np.ndarray | None:
        """Torso colour of a track's pose sample nearest `frame` (within max_df frames)."""
        if self.torso_lab is None:
            return None
        rows = np.flatnonzero((self.track_id == track_id) & (np.abs(self.frame - frame) <= max_df)
                              & np.isfinite(self.torso_lab).all(axis=1))
        if not len(rows):
            return None
        return self.torso_lab[rows[np.argmin(np.abs(self.frame[rows] - frame))]]


class LiberoColour:
    """Is a player in the libero's jersey? The libero's colour is learnt from the torsos of
    back views read as a libero number, the team's from the other numbers; a player nearer
    one by `margin` (Lab units) is that. The libero's white #6 reads poorly
    (pakmen-black_set2), so its track picked up other numbers; the colour fixes that."""

    def __init__(self, poses: Poses, read_tracks: np.ndarray, read_frames: np.ndarray,
                 read_numbers: np.ndarray, liberos: list[int], cfg: dict):
        self.ok, self.cfg, self.poses = False, cfg, poses
        if poses.torso_lab is None or not liberos or not len(read_tracks):
            return
        key = {(int(f), int(t)): i for i, (f, t) in enumerate(zip(poses.frame, poses.track_id,
                                                                 strict=True))}
        rows = [key.get((int(f), int(t))) for f, t in zip(read_frames, read_tracks, strict=True)]
        lab = np.array([poses.torso_lab[r] if r is not None else [np.nan] * 3 for r in rows])
        is_lib = np.isin(read_numbers, liberos)
        good = np.isfinite(lab).all(axis=1)
        if (good & is_lib).sum() < cfg["min_reads"] or (good & ~is_lib).sum() < cfg["min_reads"]:
            return
        self.libero = np.median(lab[good & is_lib], axis=0)
        self.team = np.median(lab[good & ~is_lib], axis=0)
        self.ok = float(np.linalg.norm(self.libero - self.team)) >= cfg["min_separation"]
        self.number = liberos[0]

    def which(self, track_id: int, frame: int, max_df: int) -> str | None:
        """'libero', 'team' or None (unsure, or no colour)."""
        if not self.ok:
            return None
        c = self.poses.colour_at(track_id, frame, max_df)
        if c is None:
            return None
        dl, dt = np.linalg.norm(c - self.libero), np.linalg.norm(c - self.team)
        if dl + self.cfg["margin"] < dt:
            return "libero"
        if dt + self.cfg["margin"] < dl:
            return "team"
        return None


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _at(poses: Poses, rows: np.ndarray, frame: float) -> tuple[np.ndarray, np.ndarray, float]:
    """One track's keypoints, scores and box height at `frame`, interpolated between the
    samples either side (rows sorted by frame)."""
    f = poses.frame[rows].astype(float)
    k = int(np.searchsorted(f, frame))
    if k == 0 or k == len(rows):
        r = rows[min(k, len(rows) - 1)]
        return poses.kp[r], poses.kp_score[r], float(poses.box[r][3] - poses.box[r][1])
    r0, r1 = rows[k - 1], rows[k]
    w = (frame - f[k - 1]) / max(f[k] - f[k - 1], 1e-9)
    kp = (1 - w) * poses.kp[r0] + w * poses.kp[r1]
    sc = np.minimum(poses.kp_score[r0], poses.kp_score[r1])
    h = (1 - w) * (poses.box[r0][3] - poses.box[r0][1]) + w * (poses.box[r1][3] - poses.box[r1][1])
    return kp, sc, float(h)


class TouchPicker:
    def __init__(self, poses: Poses, fps: float, cfg: dict):
        self.poses, self.fps, self.cfg = poses, fps, cfg
        order = np.lexsort((poses.frame, poses.track_id))
        self.by_track: dict[int, np.ndarray] = {}
        tids = poses.track_id[order]
        for t in np.unique(tids):
            self.by_track[int(t)] = order[tids == t]
        self.frames = np.sort(np.unique(poses.frame))

    def distances(self, frame: float, ball_uv) -> dict[int, float]:
        """{track: distance from the ball to its forearms / hands, in body heights}."""
        cfg = self.cfg
        if ball_uv is None or not np.isfinite(ball_uv).all():
            return {}
        p = np.asarray(ball_uv, float)
        win = cfg["max_dt_s"] * self.fps
        lo, hi = np.searchsorted(self.frames, [frame - win, frame + win + 1])
        if lo >= hi:
            return {}
        near = set(self.poses.track_id[np.isin(self.poses.frame, self.frames[lo:hi])].tolist())
        out = {}
        for tid in near:
            rows = self.by_track[int(tid)]
            rows = rows[np.abs(self.poses.frame[rows] - frame) <= win]
            if not len(rows):
                continue
            kp, sc, h = _at(self.poses, rows, frame)
            best = np.inf
            for e, w in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
                if min(sc[e], sc[w]) < cfg["keypoint_min_score"]:
                    continue
                hand = kp[w] + cfg["hand_extension"] * (kp[w] - kp[e])
                best = min(best, _segment_distance(p, kp[e], hand))
            if np.isfinite(best):
                out[int(tid)] = best / max(h, 1.0)
        return out

    def pick(self, frame: float, ball_uv) -> tuple[int, float] | None:
        """(track, confidence) of who touched the ball at `frame`, or None."""
        d = self.distances(frame, ball_uv)
        if not d:
            return None
        ranked = sorted(d.items(), key=lambda kv: kv[1])
        tid, d1 = ranked[0]
        reach = self.cfg["reach_bodies"]
        if d1 > reach:
            return None
        d2 = ranked[1][1] if len(ranked) > 1 else np.inf
        conf = min(1.0, self.cfg["base_confidence"] + (min(d2, 2 * reach) - d1) / reach)
        return tid, round(float(conf), 3)


def assign_players(rallies: list, contacts: list, picker: TouchPicker | None, numbers,
                   cfg: dict, libero: LiberoColour | None = None) -> Counter:
    """Jersey and player confidence on our touches in rallies, in place: the toucher by
    hands at the ball (else the track the touch came with, x `fallback_factor`), its number
    from `numbers` (NumberIndex). In the libero's colour: the libero; in the team's colour:
    not the libero. Kept from `min_player_confidence` on. Returns counts."""
    max_df = int(round(cfg["toucher"]["max_dt_s"] * (picker.fps if picker else 60) * 2))
    t_cfg = cfg["toucher"]
    how = Counter()
    for r in rallies:
        for i in r.contacts:
            c = contacts[i]
            if c.kind != "touch" or c.team != "us" or c.inferred:
                continue
            pick = picker.pick(c.frame, c.ball_uv) if picker is not None else None
            if pick is not None:
                tid, sure = pick
                c.track_id = tid
                how["hands at the ball"] += 1
            elif c.track_id >= 0:
                tid, sure = c.track_id, t_cfg["fallback_factor"]
                how["nearest player (no hands at the ball)"] += 1
            else:
                how["no player"] += 1
                continue
            got = numbers.number_at(tid, c.frame) if numbers is not None else None
            kit = libero.which(tid, c.frame, max_df) if libero is not None else None
            if kit == "libero":
                got = (libero.number, cfg["libero"]["confidence"])
                how["libero by jersey colour"] += 1
            elif kit == "team" and got is not None and got[0] == libero.number:
                got = None                         # read as the libero, but not in its jersey
                how["libero number but team colour"] += 1
            if got is None:
                how["no number on that track then"] += 1
                continue
            conf = round(sure * got[1], 3)
            if conf < cfg["min_player_confidence"]:
                how["number too unsure"] += 1
                continue
            c.jersey, c.player_confidence = got[0], conf
            how["number given"] += 1
    return how
