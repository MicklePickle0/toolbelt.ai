"""What each labelled touch counts as on the stat sheet (owner's rules, 2026-10-24).

Labels record what a touch looked like; stats need what it was. Our touches:
  - a dig straight after their free_ball or overpass -> free_ball_pass (an easy ball,
    not a dig)
  - a dig that ends the rally as a ball-handling error (their point; the digger lifted
    or double-contacted it) -> ball_handling_error: it wasn't controlled, so no dig.
    A clean dig followed by a called set is still a dig; the error is the setter's.
Everything else counts as labelled. Opponent touches are left as labelled. A replayed
rally (outcome `replay`) is wiped: its touches count for nothing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from setter_eyes.labels.events import ContactLabel, RallyLabel

EASY_BALLS = ("free_ball", "overpass")


def stat_actions(rally: RallyLabel) -> list[tuple[ContactLabel, str | None]]:
    out = []
    cs = rally.contacts
    for i, c in enumerate(cs):
        action = c.action
        if c.team == "us" and action == "dig":
            prev = cs[i - 1] if i else None
            if prev is not None and prev.team == "them" and prev.action in EASY_BALLS:
                action = "free_ball_pass"
            elif i == len(cs) - 1 and rally.outcome == "ball_handling_error" and \
                    rally.point_winner == "them":
                action = "ball_handling_error"
        out.append((c, action))
    return out


def counts(rallies: Iterable[RallyLabel]) -> dict[int | None, Counter]:
    """Our touches per jersey (None = jersey unknown) -> Counter of stat actions."""
    by: dict[int | None, Counter] = {}
    for r in rallies:
        if r.replayed:
            continue
        for c, action in stat_actions(r):
            if c.team == "us":
                by.setdefault(c.jersey, Counter())[action] += 1
    return by
