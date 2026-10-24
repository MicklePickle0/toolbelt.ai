"""Who is a near-side player? (Milestone 2, step 4; provisional.)

Per track, from its cleaned court positions: the share of time inside our half (with a
margin, players chase balls off court) or the far half, and how long the track lasts.
  near_player  >= `min_share` of its time in our half, lasting >= `min_duration_s`, and
               moving at least `min_move_m` around its usual spot (spectators and coaches
               standing just behind our end line don't)
  far_player   the same for the far half
  other        everyone else: referees, coaches, spectators, short fragments
Confidence = share x min(1, duration / `full_confidence_s`).

The final rule (CLAUDE.md, Milestone 4 onwards): the six people inside our half at the
serve are our players, followed through the rally wherever they run; jersey numbers and
the lineup then fix identity. This provisional version needs no serve detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from setter_eyes.track.people import PeopleTracks

ROLES = ("near_player", "far_player", "other")


def classify_tracks(tracks: PeopleTracks, fps: float, cfg: dict) -> list[dict]:
    c = cfg["roles"]
    m, net = c["court_margin_m"], c["net_overlap_m"]
    out = []
    for tid in np.unique(tracks.track_id):
        sel = tracks.track_id == tid
        xy = tracks.court_xy_smooth[sel]
        ok = np.isfinite(xy).all(axis=1)
        x, y = xy[ok, 0], xy[ok, 1]
        across = (x >= -m) & (x <= 9 + m)
        near = across & (y >= -net) & (y <= 9 + m)
        far = across & (y <= net) & (y >= -9 - m)
        n = max(1, ok.sum())
        centre = np.median(xy[ok], axis=0) if ok.any() else np.zeros(2)
        dist = np.linalg.norm(xy[ok] - centre, axis=1)
        moves = float(np.percentile(dist, 90)) if ok.any() else 0.0
        mobile = moves >= c["min_move_m"]
        fr = tracks.frame[sel]
        duration = float((fr.max() - fr.min()) / fps)
        share_near, share_far = float(near.sum() / n), float(far.sum() / n)
        if duration >= c["min_duration_s"] and share_near >= c["min_share"] and mobile:
            role, share = "near_player", share_near
        elif duration >= c["min_duration_s"] and share_far >= c["min_share"] and mobile:
            role, share = "far_player", share_far
        else:
            role, share = "other", 1 - max(share_near, share_far)
        conf = share * min(1.0, duration / c["full_confidence_s"])
        out.append({"track_id": int(tid), "role": role, "confidence": round(conf, 3),
                    "duration_s": round(duration, 2), "share_near": round(share_near, 3),
                    "share_far": round(share_far, 3), "moves_m": round(moves, 2),
                    "first_frame": int(fr.min()),
                    "last_frame": int(fr.max())})
    return out


def near_players_per_frame(tracks: PeopleTracks, roles: list[dict]) -> tuple[np.ndarray,
                                                                              np.ndarray]:
    """-> (frames, number of near-player tracks present at each)."""
    near = {r["track_id"] for r in roles if r["role"] == "near_player"}
    is_near = np.isin(tracks.track_id, list(near))
    frames = np.unique(tracks.frame)
    counts = np.bincount(np.searchsorted(frames, tracks.frame[is_near]), minlength=len(frames))
    return frames, counts


def save_roles(roles: list[dict], path: Path) -> None:
    path.write_text(json.dumps({"roles": roles}, indent=1), encoding="utf-8")


def load_roles(path: Path) -> dict[int, dict]:
    return {r["track_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))["roles"]}
