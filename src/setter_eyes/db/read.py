"""Read rallies back out of an event DB (labels DB or pipeline DB) for stats.

Stats are computed from the DB, never from video (CLAUDE.md). Rallies come back in the
same shape as parsed labels (labels.events.RallyLabel), so the stat rules in
stats/actions.py and stats/passing.py apply to labelled and reviewed data alike.
Only rallies with a point winner, or marked replay, count: an unreviewed pipeline rally
has no winner yet.
"""

from __future__ import annotations

import sqlite3

from setter_eyes.labels.events import ContactLabel, RallyLabel


def load_rallies(conn: sqlite3.Connection, video_key: str,
                 decided_only: bool = True) -> list[RallyLabel]:
    q = """SELECT r.id, s.set_number, r.rally_number, r.start_frame, r.end_frame,
                  r.point_winner, r.outcome
           FROM rallies r JOIN sets s ON r.set_id = s.id JOIN games g ON s.game_id = g.id
           WHERE g.video_key = ? ORDER BY s.set_number, r.start_t"""
    out = []
    for r in conn.execute(q, (video_key,)).fetchall():
        if decided_only and r["point_winner"] is None and r["outcome"] != "replay":
            continue
        contacts = [ContactLabel(c["frame"], c["team"], c["action"], c["jersey_number"],
                                 c["pass_score"])
                    for c in conn.execute(
                        "SELECT c.frame, c.team, c.action, c.pass_score, p.jersey_number "
                        "FROM contacts c LEFT JOIN players p ON c.player_id = p.id "
                        "WHERE c.rally_id = ? ORDER BY c.t", (r["id"],))]
        out.append(RallyLabel(r["set_number"], r["rally_number"], r["start_frame"],
                              r["end_frame"], r["point_winner"], r["outcome"], contacts))
    return out


def video_keys(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT video_key FROM games ORDER BY video_key")]
