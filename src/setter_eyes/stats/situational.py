"""Situational stats (roadmap E7/E8, owner 2026-11-16): as much as each video can give.

The owner's example: a setter who sets well in system but struggles off the net. Every
number here is a query over the event DB (search.py, the same counting as the stat sheet),
never a re-run of the vision, and every rate comes with its count: one set has ~20 passes
per player, and a % of 4 means little.

  - sets: each set of ours with the pass before it and the attack after it.
      from  where it was played: the setter's feet (contacts.player_x/y), else the pass
            spot of the receive before it (also the setter's feet)
      system  in or out of system: the pass score before it when there is one (config
            pass_scale.yaml `in_system.pass_scores`), else the setter's distance from the
            target (`in_system.max_target_distance_m`); unknown without either
      to    the zone of the attack that followed (attack_zone), and what it did:
            kill / error / in play; no attack after the set: "no attack"
    Per setter: sets, in / out of system, and the attack's kills, errors, efficiency
    in vs out of system; where the sets went.
  - passers: serve receive per player by receive zone (1, 6, 5 back row; 4, 3, 2 front),
    from the passer's feet: passes, average, errors (0s).
  - hitters: per player, attempts / kills / errors / efficiency by attack zone, in vs out of
    system, and by setter.
  - trends: per game (folders.py), the team's pass average, hitting efficiency and side-out.
Set type (tempo, 4 ball, 52, ...) isn't detected yet (roadmap G1), so it isn't a split here.
Location coverage is the limit: only touches whose toucher was found have a spot
(`located` counts say how many).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from math import hypot

from setter_eyes import folders as fold
from setter_eyes.db.read import load_rallies
from setter_eyes.search import search
from setter_eyes.stats.sheet import stat_rows

ZONES = ["4", "3", "2", "5", "6", "1"]          # front left..right, back left..right
ZONE_NAMES = {"4": "Left (4)", "3": "Middle (3)", "2": "Right (2)", "5": "Back left (5)",
              "6": "Pipe / back middle (6)", "1": "Back right (1)"}
ATTACK_LINE_M = 3.0


def court_zone(x: float | None, y: float | None, back_from_m: float = ATTACK_LINE_M) -> str | None:
    """Zone 1-6 of our half (court frame: x 0 = left sideline seen from behind our end line,
    y 0 = net). Front row in front of `back_from_m`."""
    if x is None or y is None:
        return None
    col = 0 if x < 3.0 else 1 if x < 6.0 else 2
    return ("4", "3", "2")[col] if y < back_from_m else ("5", "6", "1")[col]


def attack_zone(it: dict, back_row_m: float = 3.5) -> str | None:
    """Where an attack was hit: along the net from the ball when its 3D is known (the
    camera's reliable axis), else the hitter's feet; front or back row from the feet (depth
    is the ball's weak axis), else the ball."""
    ball, feet = it.get("ball_xy"), it.get("player_xy")
    x = ball[0] if ball else feet[0] if feet else None
    y = feet[1] if feet else ball[1] if ball else None
    if y is not None:
        y = max(y, 0.0)                     # our hitter is on our side of the net
    return court_zone(x, y, back_row_m if not feet else ATTACK_LINE_M)


def _att() -> dict:
    return {"att": 0, "k": 0, "e": 0}


def _add_att(d: dict, result: str | None) -> None:
    d["att"] += 1
    d["k"] += result == "kills"
    d["e"] += result == "attack_errors"


def _eff(d: dict) -> dict:
    return {**d, "eff": round((d["k"] - d["e"]) / d["att"], 3) if d["att"] else None,
            "kill_pct": round(d["k"] / d["att"], 3) if d["att"] else None}


def _pass() -> dict:
    return {"n": 0, "sum": 0, "errors": 0}


def _pass_out(d: dict) -> dict:
    return {"n": d["n"], "avg": round(d["sum"] / d["n"], 2) if d["n"] else None,
            "errors": d["errors"]}


def in_system(prev: dict | None, spot: list[float] | None, rule: dict,
              target: dict) -> str:
    """'in' / 'out' / 'unknown' for a set: the pass score before it, else the setter's
    distance from the target."""
    if prev is not None and prev.get("pass_score") is not None and \
            prev["action"] in ("receive", "overpass"):
        return "in" if prev["pass_score"] in rule["pass_scores"] else "out"
    if spot:
        d = hypot(spot[0] - target["x"], spot[1] - target["y"])
        return "in" if d <= rule["max_target_distance_m"] else "out"
    return "unknown"


def situational(conn: sqlite3.Connection, keys: list[str], scale: dict,
                folders: dict | None = None) -> dict:
    """scale: config/pass_scale.yaml (target, in_system, overpass_score)."""
    rule, target = scale["in_system"], scale["target"]
    items = search(conn, keys, overpass_score=scale["overpass_score"], decided_only=True)
    by_rally: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        by_rally[(it["video"], it["set_number"], it["rally_number"])].append(it)

    set_points = []
    setters: dict = defaultdict(lambda: {"n": 0, "in": _att(), "out": _att(), "unknown": _att(),
                                         "n_in": 0, "n_out": 0, "n_unknown": 0,
                                         "to": defaultdict(int)})
    hitters: dict = defaultdict(lambda: {"all": _att(), "zone": defaultdict(_att),
                                         "system": defaultdict(_att),
                                         "setter": defaultdict(_att), "located": 0})
    passers: dict = defaultdict(lambda: {"all": _pass(), "zone": defaultdict(_pass),
                                         "located": 0})
    for touches in by_rally.values():
        touches.sort(key=lambda it: it["t"])
        for i, it in enumerate(touches):
            prev = touches[i - 1] if i else None
            nxt = touches[i + 1] if i + 1 < len(touches) else None
            if it["action"] in ("receive", "overpass") and it["pass_score"] is not None:
                p = passers[it["jersey"]]
                z = court_zone(*(it["player_xy"] or (None, None)))
                for d in (p["all"], p["zone"][z or "?"]):
                    d["n"] += 1
                    d["sum"] += it["pass_score"]
                    d["errors"] += it["pass_score"] == 0
                p["located"] += z is not None
            if it["action"] == "set":
                spot = it["player_xy"] or (prev["pass_xy"] if prev and prev.get("pass_xy")
                                           else None)
                system = in_system(prev, spot, rule, target)
                attack = nxt if nxt is not None and nxt["action"] == "attack" else None
                zone = attack_zone(attack) if attack else None
                result = ("kill" if attack["result"] == "kills" else
                          "error" if attack["result"] == "attack_errors" else "in play") \
                    if attack else "no attack"
                s = setters[it["jersey"]]
                s["n"] += 1
                s[f"n_{system}"] += 1
                if attack:
                    _add_att(s[system], attack["result"])
                s["to"][zone or ("?" if attack else "no attack")] += 1
                set_points.append({
                    "setter": it["jersey"], "video": it["video"], "t": it["t"],
                    "from": [round(v, 2) for v in spot] if spot else None, "system": system,
                    "pass_score": prev["pass_score"] if prev and prev["action"] in
                    ("receive", "overpass") else None,
                    "to_zone": zone, "hitter": attack["jersey"] if attack else None,
                    "to": _to_point(attack), "result": result})
            if it["action"] == "attack":
                h = hitters[it["jersey"]]
                setter_it = prev if prev is not None and prev["action"] == "set" else None
                pre = touches[i - 2] if i >= 2 else None
                system = in_system(pre, setter_it["player_xy"] or (
                    pre["pass_xy"] if pre and pre.get("pass_xy") else None),
                    rule, target) if setter_it else "no set"
                z = attack_zone(it)
                h["located"] += z is not None
                for d in (h["all"], h["zone"][z or "?"], h["system"][system],
                          h["setter"][setter_it["jersey"] if setter_it else "no set"]):
                    _add_att(d, it["result"])

    def key(j):
        return (j is None, j if isinstance(j, int) else 0)

    return {
        "in_system_rule": {"pass_scores": rule["pass_scores"],
                           "max_target_distance_m": rule["max_target_distance_m"]},
        "target": {"x": target["x"], "y": target["y"]},
        "sets": {
            "n": len(set_points),
            "located_from": sum(p["from"] is not None for p in set_points),
            "located_to": sum(p["to"] is not None for p in set_points),
            "points": set_points,
            "by_setter": [{"jersey": j, "n": s["n"], "n_in": s["n_in"], "n_out": s["n_out"],
                           "n_unknown": s["n_unknown"], "in": _eff(s["in"]),
                           "out": _eff(s["out"]), "unknown": _eff(s["unknown"]),
                           "to": [{"zone": z, "n": s["to"][z]} for z in
                                  [*ZONES, "?", "no attack"] if s["to"].get(z)]}
                          for j, s in sorted(setters.items(), key=lambda kv: key(kv[0]))],
        },
        "passers": [{"jersey": j, **_pass_out(p["all"]), "located": p["located"],
                     "by_zone": [{"zone": z, **_pass_out(p["zone"][z])}
                                 for z in [*ZONES, "?"] if z in p["zone"]]}
                    for j, p in sorted(passers.items(), key=lambda kv: key(kv[0]))],
        "hitters": [{"jersey": j, **_eff(h["all"]), "located": h["located"],
                     "by_zone": [{"zone": z, **_eff(h["zone"][z])}
                                 for z in [*ZONES, "?"] if z in h["zone"]],
                     "by_system": [{"system": k, **_eff(h["system"][k])}
                                   for k in ("in", "out", "unknown", "no set")
                                   if k in h["system"]],
                     "by_setter": [{"setter": k, **_eff(v)} for k, v in
                                   sorted(h["setter"].items(), key=lambda kv: key(kv[0]))]}
                    for j, h in sorted(hitters.items(), key=lambda kv: key(kv[0]))],
        "trends": trends(conn, keys, scale, folders or {}),
        "zone_names": ZONE_NAMES,
    }


def _to_point(attack: dict | None) -> list[float] | None:
    """Where the attack was hit, for the map: along the net from the ball, depth from the
    hitter's feet (see attack_zone)."""
    if not attack:
        return None
    ball, feet = attack.get("ball_xy"), attack.get("player_xy")
    if not ball and not feet:
        return None
    x = ball[0] if ball else feet[0]
    y = feet[1] if feet else ball[1]
    # Our hitter hits on our side: a ball depth past the net is the 3D's weak axis.
    return [round(min(max(x, -0.5), 9.5), 2), round(min(max(y, 0.2), 9.5), 2)]


def trends(conn: sqlite3.Connection, keys: list[str], scale: dict, folders: dict) -> list[dict]:
    """Per game (its sets together), in played order when the manifest dates are set, else
    by name: the team's pass average, hitting efficiency and side-out %."""
    games: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        games[fold.folder_of(k, folders)["game"]].append(k)
    dates = {r[0]: r[1] for r in conn.execute("SELECT video_key, played_on FROM games")}
    out = []
    for game, vids in games.items():
        rallies = [r for k in vids for r in load_rallies(conn, k)]
        if not rallies:
            continue
        team = stat_rows(rallies, scale, 4)["team"]
        out.append({"game": game, "sets": len(vids), "played_on": min(
            (dates.get(k) or "" for k in vids), default=""),
            "points": f"{team['points_us']}-{team['points_them']}",
            "pass_avg": team.get("pass_avg"), "pass_n": team.get("pass_n"),
            "hitting_eff": team.get("hitting_eff"), "attacks": team.get("attacks"),
            "side_out": round(team["side_out_won"] / team["side_out_chances"], 3)
            if team.get("side_out_chances") else None,
            "side_out_n": team.get("side_out_chances")})
    out.sort(key=lambda g: (g["played_on"] or "~", g["game"]))
    return out
