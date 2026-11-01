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
The receive comes fast (owner; labelled serve -> receive 0.75-1.50 s on both sides,
receive -> set 0.83 s or more): a receive run whose first touch comes later than
`receive_max_s` after the serve has lost its receive (often the far side's, small and
hidden), so that touch is the set; later than `receive_set_max_s`, the attack. Places in
the run then start there (`missed_receive` confidence).

Likewise inside a run: two touches of one side further apart than `same_side_gap_max_s`
(labelled: consecutive touches of a side are at most 2.17 s apart; first to third 1.8-3.9
s) had a hidden touch between them, usually the set (owner: the ball goes up after their
receive, is hidden, and their side hits it). The hidden touches are returned so they can
be added as inferred touches. (A 2nd touch seen going over is NOT turned into a free
ball: usually the attack after it was missed, and that cost 4-6 sets on the labels.)

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
                   cfg: dict, launches: Sequence[float | None] | None = None,
                   serve_t: float | None = None,
                   missing_out: list | None = None) -> list[tuple[str, float]]:
    """(action, confidence) for each of `touches` (the rally's touches in time order; the
    serve touch, if seen, among them). `launches`: launch_up() per touch, for stage 2.
    serve_t: when the serve was hit if its touch wasn't seen (traced back from the ball's
    flight), for the missed-receive timing.
    missing_out: if given, gets (index of the touch it comes before, action, team, t) for
    each touch the timing says was hidden (a late first touch after the serve: the receive;
    a gap inside a side's run: usually the set), to add as inferred touches."""
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
        missed = 0                                 # leading touches of the run not seen
        hit_t = serve.t if serve is not None else serve_t
        if receive_run and hit_t is not None:
            gap = touches[first].t - hit_t
            missed = (2 if gap > cfg["receive_set_max_s"] else
                      1 if gap > cfg["receive_max_s"] else 0)
        n = len(run)
        # Two touches of one side further apart than any two in a row: one was hidden
        # between them (the set, usually: owner, 2026-10-29).
        hidden = [int(j > 0 and touches[run[j]].t - touches[run[j - 1]].t >
                      cfg["same_side_gap_max_s"]) for j in range(n)]
        before = touches[first - 1] if first > 0 else None
        after_block = (before is not None and actions[first - 1] is not None
                       and actions[first - 1][0] == "block" and before.team != team)
        last = run[-1]
        crosses = last + 1 < len(touches) and touches[last + 1].team != team
        from_end = crosses and n < 3 and not receive_run and not any(hidden)

        def place_name(place: int, receive_run=receive_run, after_block=after_block) -> str:
            if place == 1:
                return "receive" if receive_run else "coverage" if after_block else "dig"
            return _PLACE.get(place, "attack")

        offset = missed
        for j, i in enumerate(run):
            offset += hidden[j]
            place = (3 - n + 1 + j) if from_end else (j + 1 + offset)
            name = place_name(place)
            conf = (rule["uncertain"] if from_end or place > 3 else
                    rule["missed_receive"] if offset else rule[name])
            up = launches[i] if launches is not None else None
            if (name == "attack" and up is not None and np.isfinite(up)
                    and up > cfg["free_ball_min_up_mps"]):
                name, conf = "free_ball", rule["free_ball"]
            actions[i] = (name, conf)
            if missing_out is not None and hidden[j]:
                a_t, b_t = touches[run[j - 1]].t, touches[i].t
                missing_out.append((i, place_name(place - 1), team, round((a_t + b_t) / 2, 3)))
        if missing_out is not None and missed and hit_t is not None:
            t_first = touches[first].t
            est = [hit_t + cfg["receive_typical_s"]]
            if missed == 2:
                est.append(est[0] + cfg["receive_set_typical_s"])
            for p, t in enumerate(est, start=1):
                t = min(t, t_first - 0.3 * (len(est) - p + 1))
                missing_out.append((first, place_name(p), team, round(t, 3)))
        if receive_run and n == 1 and crosses and not missed:
            actions[first] = ("overpass", rule["overpass"])
        elif n == 1 and crosses and not missed and actions[first][0] == "free_ball":
            # A lofted ball sent straight back on a side's FIRST touch is an overpass (owner:
            # free balls come on the 2nd/3rd touch, overpasses only on the 1st).
            actions[first] = ("overpass", rule["overpass"])

    out = []
    for c, a in zip(touches, actions, strict=True):
        name, conf = a if a is not None else ("dig", rule["uncertain"])
        out.append((name, round(conf * (c.confidence if c.confidence else 1.0), 3)))
    return out


def fix_overpasses(rallies: list, contacts: list[Contact]) -> int:
    """An "overpass" our side plays again (our set follows, e.g. a hidden set added from the
    timing) didn't go over: it is a receive if it was our first touch after their serve,
    else a dig (owner, 2026-11-01). In place; returns how many were changed."""
    changed = 0
    for r in rallies:
        touches = [contacts[i] for i in r.contacts if contacts[i].kind == "touch"]
        for k, c in enumerate(touches[:-1]):
            if c.action != "overpass" or touches[k + 1].team != c.team:
                continue
            first = r.serving_team not in (None, c.team) and all(
                t.team != c.team for t in touches[:k])
            c.action = "receive" if first else "dig"
            changed += 1
    return changed
