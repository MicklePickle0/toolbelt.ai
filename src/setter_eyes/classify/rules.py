"""Action classification, stage 1 (CLAUDE.md step 10): rules from touch order.

After the serve, a rally's touches split into RUNS: consecutive touches by one side. A
block (a touch right at the net, |y| <= `block_zone_m`, z >= `block_min_z_m`, straight
after the other side's touch, when the ball's 3D position is known) stands alone: it
doesn't count toward that side's three, and the touches after it start a new run.

Within a run, each touch's place gives its action: 1st dig (receive in the run straight
after the opponent's serve; coverage straight after the opponent blocked this side's
ball), 2nd set, 3rd attack. Some touches are missed (about 1 in 5), so the place is
counted:
  - from the START for the receive run (its first touch is known to follow the serve)
    and for a run where the ball dies on that side;
  - from the END for a run that sends the ball over the net with fewer than three
    touches seen: the touch that sends it over is the attack, the one before it the set.
    (Counting from the start there turned attacks into digs and sets into dumps.)
A receive run of one touch that goes straight back over is an overpass, and so is any
side's single lofted touch sent straight back (overpasses only happen on a first touch).

Stage 2, from the ball's flight after the touch (`launch_up`: the vertical speed at the
start of the first confident flight after it):
  - an "attack" launched upward faster than `free_ball_min_up_mps` is a free ball: a free
    ball is lofted, an attack driven (labelled titan-rise_set1: 5 of 6 free balls above
    4 m/s up, none of 30 attacks; few examples, to confirm on more sets).
Dumps, and attack vs free ball by arm swing, need pose and aren't called yet.

Confidence: the rule's (config) times the touch's own; places counted from the end and a
4th+ touch (one of them probably wrong) get the `uncertain` confidence.
Stage 3 (a learned classifier on corrected touches) replaces these later.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from setter_eyes.ball.arcs import Arc
from setter_eyes.contacts.events import Contact

_PLACE = {1: "dig", 2: "set", 3: "attack"}


def _at_net(c: Contact, cfg: dict) -> bool:
    if c.ball_xyz is None:
        return False
    return abs(c.ball_xyz[1]) <= cfg["block_zone_m"] and c.ball_xyz[2] >= cfg["block_min_z_m"]


def launch_up(c: Contact, arcs: Sequence[Arc], cfg: dict) -> float | None:
    """Vertical speed (m/s, + up) at the start of the first confident flight leaving the
    touch, or None when no confident flight starts within `launch_window_s`."""
    for a in arcs:
        if a.confidence >= cfg["launch_min_arc_confidence"] and \
                c.t - 0.05 <= a.t0 <= c.t + cfg["launch_window_s"]:
            return float(a.v0[2])
    return None


def classify_rally(touches: Sequence[Contact], serve: Contact | None, serving_team: str | None,
                   cfg: dict, launches: Sequence[float | None] | None = None
                   ) -> list[tuple[str, float]]:
    """(action, confidence) for each of `touches` (the rally's touches in time order; the
    serve touch, if seen, among them). `launches`: launch_up() per touch, for stage 2."""
    rule = cfg["rule_confidence"]
    actions: list[tuple[str, float] | None] = [None] * len(touches)

    # Split into blocks and runs.
    runs: list[list[int]] = []
    prev_team: str | None = None
    after_serve = True
    for i, c in enumerate(touches):
        if c is serve:
            actions[i] = ("serve", rule["serve"])
            prev_team, after_serve = c.team, True
            continue
        if prev_team is not None and c.team != prev_team and not after_serve and \
                _at_net(c, cfg):
            actions[i] = ("block", rule["block"])
            runs.append([])                        # the next touches start a new run
            prev_team = c.team
            continue
        if not runs or not runs[-1] or touches[runs[-1][-1]].team != c.team:
            runs.append([i])
        else:
            runs[-1].append(i)
        prev_team, after_serve = c.team, False
    runs = [r for r in runs if r]

    for k, run in enumerate(runs):
        team = touches[run[0]].team
        first = run[0]
        receive_run = k == 0 and serving_team is not None and team != serving_team
        before = touches[first - 1] if first > 0 else None
        after_block = before is not None and actions[first - 1] is not None and \
            actions[first - 1][0] == "block" and before.team != team
        last = run[-1]
        crosses = last + 1 < len(touches) and touches[last + 1].team != team
        n = len(run)
        from_end = crosses and n < 3 and not receive_run
        for j, i in enumerate(run):
            place = (3 - n + 1 + j) if from_end else (j + 1)
            if place == 1:
                name = "receive" if receive_run else "coverage" if after_block else "dig"
            else:
                name = _PLACE.get(place, "attack")
            conf = rule["uncertain"] if from_end or place > 3 else rule[name]
            up = launches[i] if launches is not None else None
            if name == "attack" and up is not None and np.isfinite(up) and \
                    up > cfg["free_ball_min_up_mps"]:
                name, conf = "free_ball", rule["free_ball"]
            actions[i] = (name, conf)
        if receive_run and n == 1 and crosses:
            actions[first] = ("overpass", rule["overpass"])
        elif n == 1 and crosses and actions[first][0] == "free_ball":
            # A lofted ball sent straight back on a side's FIRST touch is an overpass (owner:
            # free balls come on the 2nd/3rd touch, overpasses only on the 1st).
            actions[first] = ("overpass", rule["overpass"])

    out = []
    for c, a in zip(touches, actions, strict=True):
        name, conf = a if a is not None else ("dig", rule["uncertain"])
        out.append((name, round(conf * (c.confidence if c.confidence else 1.0), 3)))
    return out
