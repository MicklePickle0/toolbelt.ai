"""Serve-receive passing: which touch is each reception, its 0-4 score, passing averages.

A reception is OUR first touch after THEIR serve (CLAUDE.md: the first ball only, not
defence). Its score, by what happened (config/pass_scale.yaml):
  - labelled `receive` with a pass_score: that score
  - labelled `overpass` (the pass went straight back over the net): `overpass_score` (0).
    Still a serve receive, so it counts in the passer's average.
  - no touch of ours and the rally ended in an ace: `ace_score` (0), passer unknown
    (counts in the team average only)
  - `receive` without a score: a reception, but unscored (left out of averages)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from setter_eyes.labels.events import RallyLabel


@dataclass(frozen=True)
class Reception:
    rally: int
    frame: int | None            # None for an untouched ace
    jersey: int | None
    score: int | None
    how: str                     # receive | overpass | ace | unscored


def receptions(rallies: Iterable[RallyLabel], scale: dict) -> list[Reception]:
    out = []
    for r in rallies:
        if r.serving_team != "them":
            continue
        serve_seen = False
        ours = None
        for c in r.contacts:
            if c.action == "serve":
                serve_seen = True
            elif serve_seen and c.team == "us" and c.action != "block":
                ours = c
                break
        if ours is None:
            if r.outcome == "ace":
                out.append(Reception(r.rally_number, None, None, scale["ace_score"], "ace"))
            continue
        if ours.action == "overpass":
            out.append(Reception(r.rally_number, ours.frame, ours.jersey,
                                 scale["overpass_score"], "overpass"))
        elif ours.pass_score is not None:
            out.append(Reception(r.rally_number, ours.frame, ours.jersey, ours.pass_score,
                                 "receive"))
        else:
            # Touched but the rally ended right there as an ace (shanked, unplayable).
            ace = r.outcome == "ace" and ours is r.contacts[-1]
            out.append(Reception(r.rally_number, ours.frame, ours.jersey,
                                 scale["ace_score"] if ace else None,
                                 "ace" if ace else "unscored"))
    return out


def passing_average(recs: Iterable[Reception]) -> dict:
    """{"team": (count, mean), jersey: (count, mean), ...} over scored receptions."""
    by: dict = {}
    for rec in recs:
        if rec.score is None:
            continue
        for key in ("team",) + ((rec.jersey,) if rec.jersey is not None else ()):
            by.setdefault(key, []).append(rec.score)
    return {k: (len(v), round(sum(v) / len(v), 2)) for k, v in by.items()}
