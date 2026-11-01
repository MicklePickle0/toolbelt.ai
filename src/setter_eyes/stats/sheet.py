"""The stat sheet (Milestone 9, first version): team and per-player stats for our team.

Built from rallies read out of the event DB (db/read.py), with the owner's counting rules
(stats/actions.py: free-ball passes and uncontrolled digs; stats/passing.py: serve
receive 0-4, overpass on serve receive = 0, aces = 0). Replays and dismissed rallies count
for nothing.

Credit for how a point ended follows the rally's outcome (labelled, or set in review)
and goes to the touch of ours it is about:
  kill (we won)               our last ball sent over    -> kills (attack, or a free ball /
                                                              overpass / dump they couldn't
                                                              control: outcome.rally_outcome)
  attack error (we lost)      our last attack            -> attack errors
  block point (we won)        our last block             -> block points
  block point (they won)      our last attack (blocked)  -> attack errors
  ace (we won)                our serve                  -> aces
  ace (they won)              our receive, if touched    -> reception errors
  service error (they won)    our serve                  -> service errors
  reception error (they won)  our receive                -> reception errors
  ball-handling error (they won)  our last touch         -> ball-handling errors
Assists: our set straight before a kill by our attack.
Hitting efficiency = (kills - attack errors) / attacks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from setter_eyes.classify.outcome import SENT_OVER
from setter_eyes.labels.events import ContactLabel, RallyLabel
from setter_eyes.stats.actions import stat_actions
from setter_eyes.stats.passing import passing_average, receptions

SKIP = ("replay", "not_a_rally")

# Stat sheet columns in order, and their short headers (scripts/stat_sheet.py, the app).
COLUMNS = ["serves", "aces", "service_errors", "attacks", "kills", "attack_errors",
           "hitting_eff", "assists", "receptions", "pass_n", "pass_avg", "reception_errors", "digs",
           "free_ball_passes", "sets", "blocks", "block_points", "ball_handling_errors"]
SHORT = {"serves": "Srv", "aces": "Ace", "service_errors": "SE", "attacks": "Att",
         "kills": "K", "attack_errors": "AE", "hitting_eff": "Eff", "assists": "Ast",
         "receptions": "Rec",
         "pass_n": "PassN", "pass_avg": "PassAvg", "reception_errors": "RE", "digs": "Dig",
         "free_ball_passes": "FBP", "sets": "Set", "blocks": "Blk",
         "block_points": "BlkPt", "ball_handling_errors": "BHE"}


_CREDIT = {  # (outcome, did we win) -> (which of our touches, stat)
    ("kill", True): (SENT_OVER, "kills"),
    ("attack_error", False): ("attack", "attack_errors"),
    ("block_point", True): ("block", "block_points"),
    ("block_point", False): ("attack", "attack_errors"),
    ("ace", True): ("serve", "aces"),
    ("ace", False): ("receive", "reception_errors"),
    ("service_error", False): ("serve", "service_errors"),
    ("reception_error", False): ("receive", "reception_errors"),
    ("ball_handling_error", False): (None, "ball_handling_errors"),
}


def _decider(r: RallyLabel) -> tuple[ContactLabel | None, str | None]:
    """(our touch credited with how the point ended, the stat it gets) or (None, None)."""
    rule = _CREDIT.get((r.outcome, r.point_winner == "us")) if r.point_winner else None
    if rule is None:
        return None, None
    action, stat = rule
    ours = [(c, a) for c, a in stat_actions(r) if c.team == "us"]
    if action is not None:
        wanted = action if isinstance(action, tuple) else (action,)
        ours = [(c, a) for c, a in ours if a in wanted]
    return (ours[-1][0], stat) if ours else (None, None)


def _assist(r: RallyLabel, killer: ContactLabel) -> ContactLabel | None:
    """Our set straight before our killing attack (the setter's assist), if any."""
    acts = stat_actions(r)
    i = next((k for k, (c, _) in enumerate(acts) if c is killer), None)
    if i is None or acts[i][1] != "attack" or i == 0:
        return None
    c, a = acts[i - 1]
    return c if c.team == "us" and a == "set" else None


def stat_rows(rallies: Iterable[RallyLabel], scale: dict, pass_top: int = 4) -> dict:
    """{"team": {...}, jersey: {...}} (jersey None = player unknown). pass_top: 4 or 3,
    the scale the passing average is read on (stats/passing.py)."""
    rallies = [r for r in rallies if r.outcome not in SKIP]
    by: dict = defaultdict(Counter)
    for r in rallies:
        for c, action in stat_actions(r):
            if c.team != "us":
                continue
            for key in ("team", c.jersey):
                if action == "serve":
                    by[key]["serves"] += 1
                elif action == "attack":
                    by[key]["attacks"] += 1
                elif action in ("dig", "free_ball_pass", "set", "block", "receive"):
                    by[key][{"dig": "digs", "free_ball_pass": "free_ball_passes",
                             "set": "sets", "block": "blocks", "receive": "receptions"}
                            [action]] += 1
        c, stat = _decider(r)
        if c is not None:
            for key in ("team", c.jersey):
                by[key][stat] += 1
            setter = _assist(r, c) if stat == "kills" else None
            if setter is not None:
                for key in ("team", setter.jersey):
                    by[key]["assists"] += 1
    team = by["team"]
    us = sum(r.point_winner == "us" for r in rallies)
    them = sum(r.point_winner == "them" for r in rallies)
    side_out = [r for r in rallies if r.serving_team == "them"]
    serving = [r for r in rallies if r.serving_team == "us"]
    team.update({"points_us": us, "points_them": them,
                 "side_out_won": sum(r.point_winner == "us" for r in side_out),
                 "side_out_chances": len(side_out),
                 "serve_points_won": sum(r.point_winner == "us" for r in serving),
                 "serve_points_played": len(serving)})
    avg = passing_average(receptions(rallies, scale), pass_top)
    out = {}
    for key, c in by.items():
        row = dict(c)
        if c["attacks"]:
            row["hitting_eff"] = round((c["kills"] - c["attack_errors"]) / c["attacks"], 3)
        if key in avg:
            row["pass_n"], row["pass_avg"] = avg[key]
        out[key] = row
    return out
