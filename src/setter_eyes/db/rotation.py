"""Write each rally's rotation, and who served our serves, into the event DB.

Reads the set's starting lineup (sets.starting_lineup, from the manifest's `lineups`) and
its rallies in time order, follows the rotation (identity/rotation.py) and writes
rallies.rotation plus player_id / player_confidence on our unreviewed serve contacts.
A reviewed serve with a player (labels, or a fix in the review UI) anchors the rotation,
so one correction fixes every rally after it. Rallies reviewed as `not_a_rally` are
skipped. Without a lineup the set is left untouched.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from setter_eyes import config
from setter_eyes.identity.rotation import follow_rotation


@dataclass(frozen=True)
class RotationSummary:
    rallies: int
    servers_set: int          # unreviewed serves given a player
    anchors: int              # reviewed serves the rotation was checked against
    flagged: int              # rallies with a substitution, resync or unknown serve


def apply_rotation(conn: sqlite3.Connection, set_id: int,
                   cfg: dict | None = None) -> RotationSummary | None:
    cfg = cfg or config.load("pipeline")
    row = conn.execute("SELECT s.starting_lineup, g.team_id FROM sets s "
                       "JOIN games g ON s.game_id = g.id WHERE s.id = ?", (set_id,)).fetchone()
    if row is None or not row[0]:
        return None
    lineup, team_id = json.loads(row[0]), row[1]
    players = dict(conn.execute("SELECT jersey_number, id FROM players WHERE team_id = ?",
                                (team_id,)).fetchall())
    jerseys = {pid: n for n, pid in players.items()}
    rallies = conn.execute(
        "SELECT id, serving_team, segment_confidence, reviewed FROM rallies "
        "WHERE set_id = ? AND (outcome IS NULL OR outcome != 'not_a_rally') ORDER BY start_t",
        (set_id,)).fetchall()
    serves = []                   # per rally: (contact id, player_id, reviewed) or None
    for r in rallies:
        serves.append(conn.execute(
            "SELECT id, player_id, reviewed FROM contacts WHERE rally_id = ? AND team = 'us' "
            "AND action = 'serve' ORDER BY t LIMIT 1", (r[0],)).fetchone())
    anchors = [jerseys.get(s[1]) if s is not None and s[2] and s[1] is not None else None
               for s in serves]
    states = follow_rotation(
        lineup, [r[1] for r in rallies], cfg,
        confidences=[1.0 if r[3] else (r[2] or 0.0) for r in rallies], servers=anchors)

    servers_set = 0
    with conn:
        for r, serve, state in zip(rallies, serves, states, strict=True):
            conn.execute("UPDATE rallies SET rotation = ? WHERE id = ?", (state.rotation, r[0]))
            if serve is None or serve[2] or state.server is None:
                continue
            if state.server not in players:     # lineup jersey with no player row yet
                players[state.server] = conn.execute(
                    "INSERT INTO players (team_id, jersey_number, name) VALUES (?, ?, ?)",
                    (team_id, state.server, f"#{state.server}")).lastrowid
            conn.execute("UPDATE contacts SET player_id = ?, player_confidence = ? WHERE id = ?",
                         (players[state.server], state.confidence, serve[0]))
            servers_set += 1
    return RotationSummary(len(rallies), servers_set, sum(a is not None for a in anchors),
                           sum(bool(s.flags) for s in states))
