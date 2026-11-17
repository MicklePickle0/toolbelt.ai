"""Game summary for the Stats page (owner, 2026-11-01): where the sets go and how the
passes are, over one set or several.

  - sets: our attacks right after a set of ours (search.py), by hitter and by zone of
    the attack (the ball's 3D position at the hit, else the hitter's feet; court metres:
    in front of `back_row_m` from the net, left / middle / right by thirds of the width;
    behind it, back row), with kills and attack errors;
  - passes: serve receives by 0-4 score, and where each was played (the setter's feet,
    stats/pass_score.py) for the pass map. An overpass on serve receive is an automatic
    0 with no spot.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict

from setter_eyes.db.read import load_rallies
from setter_eyes.search import search
from setter_eyes.stats.passing import receptions

ZONES = ["Left (4)", "Middle (3)", "Right (2)", "Back row", "Not located"]


def zone(xy: list[float] | None, back_row_m: float = 3.5) -> str:
    if not xy:
        return "Not located"
    x, y = xy
    if y > back_row_m:
        return "Back row"
    return "Left (4)" if x < 3 else "Middle (3)" if x <= 6 else "Right (2)"


def summary(conn: sqlite3.Connection, keys: list[str], scale: dict) -> dict:
    """scale: config/pass_scale.yaml."""
    overpass_score = scale["overpass_score"]
    attacks = [it for it in search(conn, keys, action="attack", overpass_score=overpass_score,
                                   decided_only=True) if it["after_set"]]
    by_hitter: dict = defaultdict(Counter)
    by_zone: dict = defaultdict(Counter)
    by_setter: Counter = Counter()
    for it in attacks:
        where = it["ball_xy"] or it["player_xy"]
        for c, k in ((by_hitter, it["jersey"]), (by_zone, zone(where))):
            c[k]["n"] += 1
            c[k]["kills"] += it["result"] == "kills"
            c[k]["errors"] += it["result"] == "attack_errors"
        by_setter[it["set_by"]] += 1
    # Passes as the stat sheet counts them (stats/passing.py receptions: rallies with a
    # winner, untouched aces as 0), so the summary and the sheet agree; spots from search.
    recs = [rec for k in keys for rec in receptions(load_rallies(conn, k), scale)
            if rec.score is not None]
    scores = Counter(rec.score for rec in recs)
    located = [it for it in search(conn, keys, action="receive", overpass_score=overpass_score,
                                   decided_only=True)
               if it["pass_score"] is not None and it["pass_xy"]]
    return {
        "sets": {
            "n": len(attacks),
            "by_hitter": sorted(({"jersey": k, **v} for k, v in by_hitter.items()),
                                key=lambda d: -d["n"]),
            "by_zone": [{"zone": z, **by_zone[z]} for z in ZONES if z in by_zone],
            "by_setter": [{"jersey": k, "n": v} for k, v in by_setter.most_common()],
        },
        "passes": {
            "n": len(recs),
            "avg": round(sum(rec.score for rec in recs) / len(recs), 2) if recs else None,
            "by_score": [{"score": s, "n": scores.get(s, 0)} for s in range(5)],
            "points": [{"x": it["pass_xy"][0], "y": it["pass_xy"][1], "score": it["pass_score"],
                        "jersey": it["jersey"], "video": it["video"], "t": it["t"]}
                       for it in located],
        },
    }
