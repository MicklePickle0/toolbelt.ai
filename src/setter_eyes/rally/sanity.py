"""Score and stat sanity checks over one set's rallies (roadmap E2, owner 2026-11-16).

Goal: hands-free. A failed check never changes a rally. It lowers the rally's review
priority (its confidence times each failed check's factor), so the doubtful rallies come
first in the review queue and the rest can be trusted. Checks run on the event DB as it is
(pipeline guesses and reviews alike), so they are always current; nothing is stored.

Set level (the match format, match.py):
  - past_end: the score reached a legal set end, yet rallies with a winner follow. Those
    rallies are flagged, and so are the least sure unreviewed ones (one per extra point:
    most likely false rallies the pipeline took for points): maybe_false.
  - short: the set ends below a legal end. Where a point is probably missing is flagged:
    a serve-order break after a long gap (missed_point), else the longest gaps
    (gap_before). One point short with the last winner unknown (no next serve) is normal:
    `last_point_unknown`.
Per rally:
  - serve_order: the winner serves next (a replay is served again by the same side). The
    serve touch says who served when it was seen, else the rally's serving team;
  - serving_team_wrong: the rally's stored serving team (side-out stats use it) disagrees
    with its serve touch. Seen on reviewed sets, where the pipeline's guess was left;
  - no_serve / two_serves / serve_not_first: one serve, first;
  - four_touches: at most 3 touches a side in a row (a block doesn't count, and the side
    that blocked gets 3 more);
  - receive_on_own_serve / receive_not_first: a receive is the first touch after the
    other side's serve;
  - outcome_side: an ace / service error belongs to the serving side; a kill needs an
    attack (or a ball sent over, or a block) by the winner; an attack error needs an attack
    by the loser.
Factors are in config/pipeline.yaml `sanity.factors`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from setter_eyes.classify.outcome import SENT_OVER
from setter_eyes.match import MatchFormat

SKIP = ("replay", "not_a_rally")

TEXT = {
    "past_end": "The set was already over at {score} (rally #{at}): this rally shouldn't count. "
                "It may be a false rally, or an earlier winner is wrong.",
    "maybe_false": "The set score passes a legal end by {extra}; this is one of the least sure "
                   "rallies, so it may be a false one.",
    "gap_before": "The set ends short at {score}, and {gap:.0f} s passed before this rally: a "
                  "point may be missing here.",
    "missed_point": "{who} won rally #{n}, but {server} serve this one, {gap:.0f} s later: a point "
                    "is probably missing in between.",
    "serve_order": "{who} won rally #{n}, but {server} serve the next one.",
    "serving_team_wrong": "The serve is by {by}{why}, but the rally says {said} served: its "
                          "serving team looks wrong (side-out and serve stats use it).",
    "no_serve": "No serve in this rally.",
    "two_serves": "{n} serves in one rally: two points may be joined.",
    "serve_not_first": "A touch comes before the serve.",
    "four_touches": "{who} touched it {n} times in a row (3 allowed, plus a block).",
    "receive_on_own_serve": "{who} received {whose} own serve.",
    "receive_not_first": "The receive isn't {whose} first touch after the serve.",
    "outcome_side": "{why}",
}


def _we(t: str | None) -> str:
    return {"us": "We", "them": "They"}.get(t or "", "?")


def _us(t: str | None) -> str:
    return {"us": "us", "them": "them"}.get(t or "", "?")


def _our(t: str | None) -> str:
    return {"us": "our", "them": "their"}.get(t or "", "?")


def _other(t: str) -> str:
    return "them" if t == "us" else "us"


@dataclass
class RallyCheck:
    id: int
    flags: list[dict] = field(default_factory=list)
    score: tuple[int, int] = (0, 0)       # running score after this rally (us, them)
    priority: float = 1.0                  # confidence x failed checks' factors (low: review)

    def add(self, code: str, factor: float, **kw) -> None:
        if self.has(code):
            return
        self.flags.append({"code": code, "text": TEXT[code].format(**kw), "factor": factor})
        self.priority *= factor

    def has(self, code: str) -> bool:
        return any(f["code"] == code for f in self.flags)


def _touches(r: dict) -> list[dict]:
    return [t for t in r["touches"] if t["team"] in ("us", "them")]


def seen_server(r: dict) -> str | None:
    """The team of the rally's first serve touch, if one was seen."""
    return next((t["team"] for t in _touches(r) if t["action"] == "serve"), None)


def _touch_flags(r: dict, add) -> None:
    """Per-rally touch checks. r: rally dict with serving_team, touches [{team, action}]."""
    touches = _touches(r)
    serves = [i for i, t in enumerate(touches) if t["action"] == "serve"]
    if not serves:
        add("no_serve")
    else:
        if len(serves) > 1:
            add("two_serves", n=len(serves))
        if serves[0] > 0:
            add("serve_not_first")
    side, run = None, 0
    for t in touches:
        if t["action"] == "block":
            side, run = t["team"], 0
            continue
        run = run + 1 if t["team"] == side else 1
        side = t["team"]
        if run > 3:
            add("four_touches", who=_we(side), n=run)
            break
    server = touches[serves[0]]["team"] if serves else r["serving_team"]
    for i, t in enumerate(touches):
        if t["action"] != "receive":
            continue
        if server and t["team"] == server:
            add("receive_on_own_serve", who=_we(t["team"]), whose=_our(t["team"]))
        elif serves and any(u["team"] == t["team"] for u in touches[serves[0] + 1:i]):
            add("receive_not_first", whose=_our(t["team"]))


def _outcome_flags(r: dict, add, server: str | None) -> None:
    out, win = r["outcome"], r["point_winner"]
    if win is None or out is None:
        return
    lose = _other(win)
    acts = {(t["team"], t["action"]) for t in _touches(r)}
    if out == "ace" and server and win != server:
        add("outcome_side", why=f"An ace, but the point went to {_us(win)}, who didn't serve.")
    elif out == "service_error" and server and win == server:
        add("outcome_side", why=f"A service error, but the serving side ({_us(win)}) won it.")
    elif out == "kill" and not any((win, a) in acts for a in (*SENT_OVER, "block")):
        add("outcome_side", why=f"A kill for {_us(win)}, but there's no attack or ball sent over "
                                f"by {_us(win)} in this rally.")
    elif out == "attack_error" and not any((lose, a) in acts for a in SENT_OVER):
        add("outcome_side", why=f"An attack error by {_us(lose)}, but there's no attack by "
                                f"{_us(lose)} in this rally.")


def check_set(rallies: list[dict], fmt: MatchFormat, set_number: int, cfg: dict) -> dict:
    """rallies: time order, each {id, rally_number, start_t, end_t, segment_confidence,
    serving_team, point_winner, outcome, reviewed, touches: [{team, action}]}.
    cfg: config/pipeline.yaml. Returns {status, message, score, target, format, set_number,
    rallies: {id: {flags, score, priority}}, flagged: [ids, least priority first],
    reviewed_flagged}."""
    fac = cfg["sanity"]["factors"]
    gap_s = cfg["rally"].get("missed_point_gap_s", 20.0)
    checks = {r["id"]: RallyCheck(r["id"], priority=1.0 if r["reviewed"] else
                                  float(r["segment_confidence"] or 0.0)) for r in rallies}

    def adder(rid):
        return lambda code, **kw: checks[rid].add(code, fac[code], **kw)

    counted = [r for r in rallies if r["outcome"] not in SKIP]
    # Who served each rally: the serve touch when seen, else the rally's serving team.
    server = {r["id"]: seen_server(r) or r["serving_team"] for r in rallies}
    for r in counted:
        _touch_flags(r, adder(r["id"]))
        _outcome_flags(r, adder(r["id"]), server[r["id"]])

    # The serving team stored on the rally (side-out stats use it) vs its serve touch.
    prev_winner = None
    for r in counted:
        seen = seen_server(r)
        if seen and r["serving_team"] and seen != r["serving_team"]:
            why = f" and the point before went to {_us(seen)}" if prev_winner == seen else ""
            said = "we" if r["serving_team"] == "us" else "they"
            adder(r["id"])("serving_team_wrong", by=_us(seen), why=why, said=said)
        prev_winner = r["point_winner"]

    # The winner serves next (replays aren't counted: the same side serves again).
    for r, nxt in zip(counted, counted[1:], strict=False):
        expect, got = r["point_winner"], server[nxt["id"]]
        if not expect or not got or expect == got:
            continue
        gap = nxt["start_t"] - r["end_t"]
        kw = {"who": _we(expect), "n": r["rally_number"], "server": _we(got).lower()}
        if gap > gap_s:
            adder(nxt["id"])("missed_point", gap=gap, **kw)
        else:
            adder(r["id"])("serve_order", **kw)
            adder(nxt["id"])("serve_order", **kw)

    # Score and set end.
    us = them = 0
    ended_at = None
    for r in counted:
        if ended_at is not None and r["point_winner"]:
            adder(r["id"])("past_end", score=f"{ended_at[1]}-{ended_at[2]}", at=ended_at[0])
        us += r["point_winner"] == "us"
        them += r["point_winner"] == "them"
        checks[r["id"]].score = (us, them)
        if ended_at is None and fmt.set_over(us, them, set_number):
            ended_at = (r["rally_number"], us, them)
    for r in rallies:
        if r["outcome"] in SKIP:
            prev = [checks[x["id"]].score for x in counted if x["start_t"] < r["start_t"]]
            checks[r["id"]].score = prev[-1] if prev else (0, 0)

    target = fmt.target(set_number)
    score = f"{us}-{them}"
    if ended_at is not None and (us, them) != (ended_at[1], ended_at[2]):
        extra = (us + them) - (ended_at[1] + ended_at[2])
        status = "past_end"
        message = (f"Score {score} passes a legal end ({ended_at[1]}-{ended_at[2]} at rally "
                   f"#{ended_at[0]}; set {set_number} is to {target}, win by {fmt.win_by}). "
                   f"{extra} point(s) too many: false rallies or wrong winners.")

        def doubt(r):                          # least sure so far; hardly any touches: more
            few = 0.5 if len(_touches(r)) <= 1 else 1.0
            return (checks[r["id"]].priority * few, r["start_t"])
        unsure = sorted((r for r in counted if not r["reviewed"] and r["point_winner"]),
                        key=doubt)
        for r in unsure[:extra]:
            adder(r["id"])("maybe_false", extra=extra)
    elif ended_at is not None:
        status, message = "ok", f"Score {score}: a legal end of set {set_number} (to {target})."
    else:
        last = counted[-1] if counted else None
        one_more = last is not None and last["point_winner"] is None and (
            fmt.set_over(us + 1, them, set_number) or fmt.set_over(us, them + 1, set_number))
        if one_more:
            status = "last_point_unknown"
            message = (f"Score {score} before the last rally, whose winner isn't known (no next "
                       "serve): set it in review to finish the set.")
        else:
            status = "short"
            need = max(target - max(us, them), fmt.win_by - abs(us - them), 1)
            message = (f"Score {score}: set {set_number} is to {target}, win by {fmt.win_by}, "
                       f"so at least {need} point(s) are missing (missed rallies, or the video "
                       "ends early).")
            # Where: serve-order breaks after a long gap first (missed_point above), then the
            # longest other gaps, a couple more than the points missing.
            found = sum(checks[r["id"]].has("missed_point") for r in counted)
            gaps = sorted(((r["start_t"] - prev["end_t"], r) for r, prev in
                           zip(counted[1:], counted, strict=False)
                           if r["start_t"] - prev["end_t"] > gap_s
                           and not checks[r["id"]].has("missed_point")),
                          key=lambda g: -g[0])
            for gap, r in gaps[:max(0, need + 2 - found)]:
                adder(r["id"])("gap_before", score=score, gap=gap)

    rallies_out = {rid: {"flags": c.flags, "score": list(c.score),
                         "priority": round(c.priority, 4)} for rid, c in checks.items()}
    flagged = sorted((rid for rid, c in checks.items() if c.flags),
                     key=lambda rid: checks[rid].priority)
    reviewed = {r["id"] for r in rallies if r["reviewed"]}
    return {"status": status, "message": message, "score": [us, them], "target": target,
            "set_number": set_number, "format": fmt.to_dict(), "rallies": rallies_out,
            "flagged": flagged, "reviewed_flagged": [rid for rid in flagged if rid in reviewed]}


def load_set(conn: sqlite3.Connection, set_id: int) -> list[dict]:
    """A set's rallies with their touches, for check_set()."""
    rows = [dict(r) for r in conn.execute(
        "SELECT id, rally_number, start_t, end_t, segment_confidence, serving_team, "
        "point_winner, point_winner_source, outcome, reviewed FROM rallies WHERE set_id = ? "
        "ORDER BY start_t", (set_id,))]
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        r["touches"] = []
    for c in conn.execute(
            "SELECT c.rally_id, c.team, c.action FROM contacts c JOIN rallies r "
            "ON c.rally_id = r.id WHERE r.set_id = ? ORDER BY c.t", (set_id,)):
        by_id[c[0]]["touches"].append({"team": c[1], "action": c[2]})
    return rows
