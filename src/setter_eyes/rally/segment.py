"""Rally segmentation and serve detection (Milestone 4). No audio; offline.

A rally is a stretch where the ball is followed continuously (no gap over `max_ball_gap_s`)
and crosses the net at least once. It starts at the serve, if found near its beginning,
else at the first ball sample, and ends at the first floor contact after the serve, else
where the ball is lost.

Serve, either of:
  - a touch with the ball at or behind an end line (|y| >= `end_line_y_m`), at a hitting
    height, followed by a flight that crosses the net;
  - the rally's first net-crossing flight traced back (it's ballistic) to just behind an
    end line lands at a hitting height with serve speed. Serves often aren't tracked from
    the hit (blur right in front of the camera, a tiny ball 25 m away), so this catches the
    ones whose touch was never seen.
Behind our end line (y > 0) we serve, behind theirs they do.

Rally-state prior (CLAUDE.md): at most 3 touches a side in a row (plus a block); a 4th
lowers that touch's confidence rather than being dropped.

With the learned touch scorer, rallies are built from touches instead
(segment_from_touches): touches less than `max_touch_gap_s` apart are one rally (labelled
footage: <= 2.2 s apart within a rally, >= 14 s between rallies). The serve, in order:
  - traced back from the first net-crossing flight, when it starts before the first
    touch seen and from the other side (the hit wasn't seen, the reply was)
  - the last touch that looks like a serve (at or behind an end line, then over the net)
    before the ball first changes sides
  - a single touch followed by the ball changing sides (a serve whose 3D was weak)
  - otherwise INFERRED: a rally that opens with 2+ touches by one side (receive, set,
    attack), or only one side's touches and a net crossing, was served by the other side,
    unseen (hidden by a player or the net tape); placed `serve_flight_s` before the first
    touch
    Right about half the time on labelled footage (mostly when a touch's team is wrong;
    pre-serve bounces within `max_touch_gap_s` of the serve look the same), so these
    rallies get `inferred_factor` confidence and go to the review queue.
A group with none of these is dropped (pepper in a timeout, a toss, a ball kicked back), and
so is an entry serve alone whose ball is lost at once (no serve flight: a ball held up in
front of the camera).
Touches with the ball outside our court's airspace belong to another court and are
ignored. The rally ends at the first floor contact after its last touch, else where the
ball is lost, at most `end_window_s` after it. Touches after the ball touched the floor are
dead (picked up after the point), as are trailing touches that fail the live-ball check
(rally/live.py). Often nothing visible ends a point (a net touch, a call): then the rally
just stops, without a floor contact, at lower confidence, and goes to review.

The score is NOT decided here (CLAUDE.md): rallies carry no point winner; that comes from
the review UI key press, with scoreboard OCR as a sanity check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath
from setter_eyes.contacts.events import Contact
from setter_eyes.rally.live import PlayerMotion, dead_from

SERVE_FROM_FLIGHT = -2          # Rally.serve_contact when the serve was traced back
SERVE_INFERRED = -3             # ...when it wasn't seen at all but the rally shows it
TEAMS = ("us", "them")


@dataclass
class Rally:
    index: int
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    serving_team: str | None
    serve_contact: int | None            # index into the contacts list
    end_reason: str                      # floor | ball_lost
    contacts: list[int] = field(default_factory=list)
    net_crossings: int = 0
    confidence: float = 0.0
    # From rally/winner.py (who serves next), once all rallies are known:
    point_winner: str | None = None
    winner_confidence: float = 0.0
    score_us: int = 0
    score_them: int = 0
    outcome: str | None = None
    outcome_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def net_crossings(arcs: list[Arc], g: np.ndarray, t0: float, t1: float,
                  min_conf: float) -> int:
    """How many times confident flights within [t0, t1] cross the net plane (y = 0)."""
    n = 0
    for a in arcs:
        if a.confidence < min_conf or a.t0 + a.duration_s < t0 or a.t0 > t1:
            continue
        ts = np.linspace(a.t0, a.t0 + a.duration_s, 20)
        y = a.position(ts, g)[:, 1]
        n += int(np.sum(np.diff(np.sign(y)) != 0))
    return n


def is_serve(c: Contact, arcs: list[Arc], g: np.ndarray, cfg: dict) -> bool:
    s = cfg["serve"]
    if c.kind != "touch" or c.ball_xyz is None:
        return False
    x, y, z = c.ball_xyz
    if abs(y) < s["end_line_y_m"] or not (s["height_m"][0] <= z <= s["height_m"][1]):
        return False
    nxt = [a for a in arcs if a.first >= c.path_index and a.confidence >= s["min_arc_confidence"]]
    if not nxt:
        return False
    a = nxt[0]
    ys = a.position(np.linspace(a.t0, a.t0 + a.duration_s, 20), g)[:, 1]
    return bool(np.sign(ys[0]) != np.sign(ys[-1]) or np.sign(ys[-1]) != np.sign(y))


def segment_rallies(path: BallPath, arcs: list[Arc], contacts: list[Contact], g: np.ndarray,
                    fps: float, cfg: dict) -> list[Rally]:
    r = cfg["rally"]
    times = path.frames / fps
    seen = np.isfinite(path.uv).all(axis=1)
    idx = np.flatnonzero(seen)
    if not len(idx):
        return []
    # Episodes of continuous ball presence.
    breaks = np.flatnonzero(np.diff(times[idx]) > r["max_ball_gap_s"])
    episodes = np.split(idx, breaks + 1)
    for c in contacts:
        c.serve = is_serve(c, arcs, g, cfg)
    rallies: list[Rally] = []
    for ep in episodes:
        t0, t1 = float(times[ep[0]]), float(times[ep[-1]])
        if t1 - t0 < r["min_rally_s"]:
            continue
        in_ep = [i for i, c in enumerate(contacts) if t0 <= c.t <= t1]
        serves = [i for i in in_ep
                  if contacts[i].serve and contacts[i].t - t0 <= r["serve_window_s"]]
        start_i = serves[0] if serves else None
        start_t = contacts[start_i].t if start_i is not None else t0
        serving = contacts[start_i].team if start_i is not None else None
        if start_i is None:
            traced = traced_serve(arcs, g, t0, t1, cfg)
            if traced is not None:
                start_t, serving, _ = traced
                start_i = SERVE_FROM_FLIGHT
        floors = [i for i in in_ep if contacts[i].kind == "floor" and contacts[i].t > start_t]
        end_t = contacts[floors[0]].t if floors else t1
        crossings = net_crossings(arcs, g, start_t, end_t, r["min_arc_confidence"])
        if crossings == 0:
            continue                                     # warm-up, a toss, a ball rolled back
        members = [i for i in in_ep if start_t <= contacts[i].t <= end_t]
        _apply_touch_prior(contacts, members, cfg, serving)
        conf = r["base_confidence"] + r["serve_bonus"] * (start_i is not None) + \
            r["floor_bonus"] * bool(floors)
        rallies.append(Rally(
            len(rallies) + 1, int(round(start_t * fps)), int(round(end_t * fps)),
            round(start_t, 3), round(end_t, 3), serving, start_i,
            "floor" if floors else "ball_lost", members, crossings, round(min(1.0, conf), 3)))
    return rallies


def in_airspace(c: Contact, cfg: dict) -> bool:
    """Over our court (see rally.airspace)? Unknown (no 3D) counts as yes."""
    if c.ball_xyz is None:
        return True
    a = cfg["rally"]["airspace"]
    x, y, z = c.ball_xyz
    return a["x_m"][0] <= x <= a["x_m"][1] and a["y_m"][0] <= y <= a["y_m"][1] and \
        z <= a["max_z_m"]


def segment_from_touches(path: BallPath, arcs: list[Arc], contacts: list[Contact],
                         g: np.ndarray, fps: float, cfg: dict,
                         motion: PlayerMotion | None = None) -> list[Rally]:
    """motion: near-side player movement, for the live-ball check (rally/live.py) that
    ends a rally before touches made after the whistle."""
    r = cfg["rally"]["from_touches"]
    times = path.frames / fps
    seen_t = times[np.isfinite(path.uv).all(axis=1)]
    for c in contacts:
        c.serve = False
    touches = [i for i, c in enumerate(contacts) if c.kind == "touch" and in_airspace(c, cfg)]
    groups: list[list[int]] = []
    for i in touches:
        # Our serve seen entering the frame always starts a rally (pre-serve bounces and
        # tosses stay with the dead time before it).
        if groups and not contacts[i].entry_serve and \
                contacts[i].t - contacts[groups[-1][-1]].t <= r["max_touch_gap_s"]:
            groups[-1].append(i)
        else:
            groups.append([i])

    rallies: list[Rally] = []
    for grp in groups:
        teams = [contacts[i].team for i in grp]
        change = next((j for j in range(len(grp) - 1)
                       if teams[j] and teams[j + 1] and teams[j] != teams[j + 1]), None)
        first_run = grp[:change + 1] if change is not None else grp
        t_first = contacts[grp[0]].t
        traced = traced_serve(arcs, g, t_first - r["serve_lookback_s"], contacts[grp[-1]].t,
                              cfg)
        crosses = net_crossings(arcs, g, t_first, contacts[grp[-1]].t + r["end_window_s"],
                                cfg["rally"]["min_arc_confidence"]) > 0
        serve_like = [i for i in first_run if is_serve(contacts[i], arcs, g, cfg)]
        other = {"us": "them", "them": "us"}.get(teams[0])
        entry = [i for i in first_run if contacts[i].entry_serve]
        serve_i: int | None = None
        if entry:
            serve_i = entry[-1]                    # our serve, seen entering the frame
        elif traced is not None and traced[0] < t_first and traced[1] != teams[0]:
            serve_i = SERVE_FROM_FLIGHT            # the hit wasn't seen; the reply was
        elif serve_like:
            serve_i = serve_like[-1]
        elif len(first_run) == 1 and (change is not None or crosses):
            serve_i = first_run[0]
        elif other is not None and (change is not None or crosses):
            serve_i = SERVE_INFERRED               # receive/set/attack first: theirs served
        elif traced is not None:
            serve_i = SERVE_FROM_FLIGHT
        if serve_i is None:
            continue                               # no serve, no exchange
        if serve_i == SERVE_FROM_FLIGHT:
            start_t, serving = traced[0], traced[1]
        elif serve_i == SERVE_INFERRED:
            start_t, serving = t_first - r["serve_flight_s"], other
        else:
            start_t, serving = contacts[serve_i].t, contacts[serve_i].team
            contacts[serve_i].serve = True
        play = [i for i in grp if contacts[i].t >= start_t]
        cut = dead_from([contacts[i].t for i in play], start_t, motion,
                        cfg["rally"]["live_ball"])
        dead_t = contacts[play[cut]].t if cut < len(play) else None
        landed = _landed_before(contacts, play)            # the ball can't bounce in play
        if landed < cut:
            cut, dead_t = landed, None
        last = contacts[play[cut - 1]].t if play[:cut] else contacts[grp[-1]].t
        stop = min(last + r["end_window_s"], dead_t if dead_t is not None else np.inf)
        floors = [i for i, c in enumerate(contacts)
                  if c.kind == "floor" and last < c.t <= stop]
        if floors:
            end_t, reason = contacts[floors[0]].t, "floor"
        else:
            end_t, reason = _ball_lost_after(seen_t, last, cfg["rally"]["max_ball_gap_s"],
                                             r["end_window_s"]), "ball_lost"
            if dead_t is not None and end_t >= dead_t:
                end_t, reason = last + 0.5 * (dead_t - last), "dead_ball"
        crossed = net_crossings(arcs, g, start_t, end_t, cfg["rally"]["min_arc_confidence"])
        if entry and len(play[:cut]) == 1 and not floors and not crossed and \
                end_t - start_t < cfg["rally"]["our_serve"]["min_flight_s"]:
            continue                               # big ball at the top, then no flight
        members = [i for i, c in enumerate(contacts) if start_t <= c.t <= end_t]
        _apply_touch_prior(contacts, members, cfg, serving)
        inferred = serve_i == SERVE_INFERRED
        conf = r["base_confidence"] + r["serve_bonus"] * (not inferred) + \
            r["exchange_bonus"] * (change is not None) + r["floor_bonus"] * bool(floors)
        if inferred:
            conf *= r["inferred_factor"]
        if dead_t is not None:
            conf *= cfg["rally"]["live_ball"]["trimmed_factor"]
        rallies.append(Rally(
            len(rallies) + 1, int(round(start_t * fps)), int(round(end_t * fps)),
            round(start_t, 3), round(end_t, 3), serving, serve_i, reason, members, crossed,
            round(min(1.0, conf), 3)))
    return rallies


def _landed_before(contacts: list[Contact], play: list[int]) -> int:
    """Index into `play` of the first touch made after the ball touched the floor (len(play)
    if none): it was picked up after the point. Labelled: 3 dead touches and 1 false one
    came after a floor contact, no real touch did."""
    floor_t = [c.t for c in contacts if c.kind == "floor"]
    for n in range(1, len(play)):
        t0, t1 = contacts[play[n - 1]].t, contacts[play[n]].t
        if any(t0 < f < t1 for f in floor_t):
            return n
    return len(play)


def _ball_lost_after(seen_t: np.ndarray, t: float, max_gap: float, cap: float) -> float:
    """When the ball is lost (no sample for > max_gap) after time t, at most t + cap."""
    last = t
    for x in seen_t[(seen_t > t) & (seen_t <= t + cap)]:
        if x - last > max_gap:
            break
        last = float(x)
    return last


def traced_serve(arcs: list[Arc], g: np.ndarray, t0: float, t1: float,
                 cfg: dict) -> tuple[float, str, list[float]] | None:
    """The first confident net-crossing flight in [t0, t1], traced back to just behind an
    end line: (serve time, serving team, ball position) if it was at hitting height and
    serve speed there, else None."""
    s = cfg["serve"]
    for a in arcs:
        if a.confidence < s["min_arc_confidence"] or a.t0 + a.duration_s < t0 or a.t0 > t1:
            continue
        ys = a.position(np.linspace(a.t0, a.t0 + a.duration_s, 20), g)[:, 1]
        if np.sign(ys[0]) == np.sign(ys[-1]):
            continue
        p0, v0 = np.array(a.p0), np.array(a.v0)
        side = np.sign(ys[0]) or 1.0                       # the end it came from
        if v0[1] * side >= 0 or np.hypot(*v0[:2]) < s["min_speed_mps"]:
            return None                                    # not travelling away from that end
        dt = (side * s["traced_from_y_m"] - p0[1]) / v0[1]  # negative: back in time
        if dt > 0 or -dt > s["max_trace_back_s"]:
            return None
        p = a.position(np.array([a.t0 + dt]), g)[0]
        lo, hi = s["traced_height_m"]
        if not lo <= p[2] <= hi:
            return None
        return float(a.t0 + dt), "us" if side > 0 else "them", p.round(3).tolist()
    return None


def _apply_touch_prior(contacts: list[Contact], members: list[int], cfg: dict,
                       serving: str | None = None) -> None:
    """At most `max_contacts_per_side` touches by one side in a row, plus a block (owner,
    2026-10-29). Each touch's team is a guess on its own (the scorer's p_ours); here the
    rally's team sequence is decoded as a whole (Viterbi): the most likely teams under that
    rule. A 4th touch in a row costs `fourth_touch_cost` (it needs a block first, and blocks
    aren't classified yet); a 5th is impossible, and so is the serving side touching right
    after its serve. A touch whose team changes keeps `over_limit_penalty` of its confidence
    (review queue)."""
    c_cfg = cfg["contacts"]
    td = c_cfg["team_decoder"]
    limit = c_cfg["max_contacts_per_side"] + 1               # +1: allow one block
    seq = [i for i in members if contacts[i].kind == "touch" and not contacts[i].serve
           and contacts[i].team in TEAMS]
    if not seq:
        return
    lo = td["p_clip"]
    cost = []                                                # per touch: {team: -log p}
    for i in seq:
        c = contacts[i]
        p = c.p_ours if c.p_ours is not None else \
            (td["unscored_p"] if c.team == "us" else 1 - td["unscored_p"])
        p = min(max(p, lo), 1 - lo)
        cost.append({"us": -np.log(p), "them": -np.log(1 - p)})
    # State (team, touches in a row). The serve counts as a full run for the serving side.
    best: dict[tuple[str, int], tuple[float, list[str]]] = {}
    start = {(serving, limit): (0.0, [])} if serving in TEAMS else {None: (0.0, [])}
    for k in range(len(seq)):
        best = {}
        for state, (total, path) in start.items():
            for team in TEAMS:
                run = state[1] + 1 if state is not None and state[0] == team else 1
                if run > limit:
                    continue
                step = cost[k][team] + (td["fourth_touch_cost"] if run == limit else 0.0)
                key = (team, run)
                if key not in best or total + step < best[key][0]:
                    best[key] = (total + step, [*path, team])
        start = best
    teams = min(best.values(), key=lambda v: v[0])[1]
    for i, team in zip(seq, teams, strict=True):
        c = contacts[i]
        if c.team != team:
            c.team = team
            c.confidence = round(c.confidence * c_cfg["over_limit_penalty"], 3)


def merge_close(rallies: list[Rally], contacts: list[Contact], cfg: dict) -> list[Rally]:
    """One point cut in two: rallies closer than `rally.merge_gap_s` (owner: there are
    usually 10+ s between points) become one. The first keeps its serve; the second's
    "serve" was a touch in play (actions are classified afterwards, from the merged rally).
    The merged rally's confidence is the lower of the two (it goes to review sooner).
    A rally that starts with a clearly seen serve is never merged in: that is a new point,
    and a false rally in the downtime would otherwise chain two real points together
    (labelled: points are always 10.6+ s apart)."""
    gap = cfg["rally"]["merge_gap_s"]
    sure = cfg["rally"]["merge_keeps_serve_conf"]
    out: list[Rally] = []
    for r in rallies:
        prev = out[-1] if out else None
        serve = (contacts[r.serve_contact] if r.serve_contact is not None
                 and r.serve_contact >= 0 else None)
        new_point = serve is not None and (serve.entry_serve or serve.confidence >= sure)
        if prev is not None and r.start_t - prev.end_t < gap and not new_point:
            prev_seen = prev.serve_contact is not None and prev.serve_contact >= 0
            if not prev_seen and serve is not None:
                out[-1] = r                     # no serve, then a serve: downtime, then a point
                continue
            if r.serve_contact is not None and r.serve_contact >= 0:
                contacts[r.serve_contact].serve = False
                contacts[r.serve_contact].entry_serve = False
            prev.contacts = sorted(set(prev.contacts) | set(r.contacts))
            prev.end_t, prev.end_frame = max(prev.end_t, r.end_t), max(prev.end_frame, r.end_frame)
            prev.end_reason = r.end_reason
            prev.net_crossings += r.net_crossings
            prev.confidence = round(min(prev.confidence, r.confidence), 3)
            continue
        out.append(r)
    for i, r in enumerate(out, start=1):
        r.index = i
    return out


def flag_gaps(rallies: list[Rally], cfg: dict) -> int:
    """A long gap before a rally may hide a point the pipeline missed: that rally's
    confidence drops (`rally.missed_point_factor`) so it comes up early in review, where
    the missed point can be added. Returns how many were flagged."""
    gap, factor = cfg["rally"]["missed_point_gap_s"], cfg["rally"]["missed_point_factor"]
    n = 0
    for prev, r in zip(rallies, rallies[1:], strict=False):
        if r.start_t - prev.end_t > gap:
            r.confidence = round(r.confidence * factor, 3)
            n += 1
    return n
