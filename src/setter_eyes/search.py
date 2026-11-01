"""Search the event DB's points (owner, 2026-11-01): every touch of ours matching filters.

Filters (all optional, combined with AND):
  - player: the touch's jersey
  - action: what the touch counts as on the stat sheet (stats/actions.py: serve, receive,
    set, attack, dig, block, free_ball_pass, ...)
  - result: the stat the touch was credited with for how the point ended (stats/sheet.py
    `_decider`: kills, aces, attack_errors, service_errors, reception_errors, ...)
  - pass_scores: the serve receive's 0-4 score (an overpass on serve receive is an
    automatic 0, as in the passing average)
  - set_by: the touch came straight after a set of ours by this jersey; with action
    `attack` and a player, "points where #17 set to #12"
The same counting as the stat sheet, so a search and the stats agree. Replays and false
rallies count for nothing.
"""

from __future__ import annotations

import sqlite3

from setter_eyes.db.read import load_rallies
from setter_eyes.stats.actions import stat_actions
from setter_eyes.stats.sheet import SKIP, _decider

RESULTS = ["kills", "aces", "attack_errors", "service_errors", "block_points",
           "reception_errors", "ball_handling_errors"]


def _rally_ids(conn: sqlite3.Connection, key: str) -> dict[tuple[int, int], dict]:
    q = """SELECT r.id, s.set_number, r.rally_number, r.start_t, r.end_t, r.reviewed
           FROM rallies r JOIN sets s ON r.set_id = s.id JOIN games g ON s.game_id = g.id
           WHERE g.video_key = ?"""
    return {(r[1], r[2]): {"rally_id": r[0], "start_t": r[3], "end_t": r[4],
                           "reviewed": bool(r[5])} for r in conn.execute(q, (key,))}


def search(conn: sqlite3.Connection, keys: list[str], player: int | None = None,
           action: str | None = None, result: str | None = None,
           pass_scores: list[int] | None = None, set_by: int | None = None,
           overpass_score: int | None = 0, decided_only: bool = False) -> list[dict]:
    """decided_only: only rallies with a point winner (as the stat sheet counts them)."""
    out = []
    for key in keys:
        fps_row = conn.execute("SELECT fps FROM games WHERE video_key = ?", (key,)).fetchone()
        if fps_row is None:
            continue
        fps = fps_row[0]
        ids = _rally_ids(conn, key)
        where = {r[0]: r[1:] for r in conn.execute(
            "SELECT c.frame, c.ball_x, c.ball_y, c.pass_x, c.pass_y, c.player_x, c.player_y "
            "FROM contacts c JOIN rallies "
            "r ON c.rally_id = r.id JOIN sets s ON r.set_id = s.id JOIN games g ON "
            "s.game_id = g.id WHERE g.video_key = ? AND c.team = 'us'", (key,))}
        for r in load_rallies(conn, key, decided_only=decided_only):
            if r.outcome in SKIP:
                continue
            decider, stat = _decider(r)
            acts = stat_actions(r)
            for i, (c, a) in enumerate(acts):
                if c.team != "us":
                    continue
                score = c.pass_score if a == "receive" else \
                    overpass_score if a == "overpass" and r.serving_team == "them" else None
                prev = acts[i - 1] if i else None
                after_set = prev is not None and prev[0].team == "us" and prev[1] == "set"
                setter = prev[0].jersey if after_set else None
                credited = stat if c is decider else None
                if player is not None and c.jersey != player:
                    continue
                if action is not None and a != action:
                    continue
                if result is not None and credited != result:
                    continue
                if pass_scores is not None and score not in pass_scores:
                    continue
                if set_by is not None and setter != set_by:
                    continue
                meta = ids.get((r.set_number, r.rally_number), {})
                bx, by, px, py, fx, fy = where.get(c.frame, (None,) * 6)
                out.append({"video": key, "set_number": r.set_number,
                            "rally_number": r.rally_number,
                            "rally_id": meta.get("rally_id"), "reviewed": meta.get("reviewed"),
                            "t": round(c.frame / fps, 3), "start_t": meta.get("start_t"),
                            "end_t": meta.get("end_t"), "jersey": c.jersey, "action": a,
                            "pass_score": score, "result": credited, "set_by": setter,
                            "after_set": after_set,
                            "point_winner": r.point_winner, "outcome": r.outcome,
                            "ball_xy": [bx, by] if bx is not None else None,
                            "pass_xy": [px, py] if px is not None else None,
                            "player_xy": [fx, fy] if fx is not None else None})
    return out
