"""Write pipeline output (rallies, touches, actions) into the event DB.

The pipeline DB (paths.db_path) has the same schema as the labels DB, so stats queries
and evaluation work on either. Everything written here is automated: rows are
unreviewed and carry their confidences; point_winner and outcome stay empty until the
review UI sets the winner (CLAUDE.md: the pipeline never decides the score).
Re-writing a video replaces everything previously written for it.

A video is one set (set 1 unless the manifest's lineups say otherwise).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from setter_eyes.labels.importer import delete_game, upsert_roster
from setter_eyes.labels.manifest import Manifest, VideoEntry


@dataclass(frozen=True)
class WriteSummary:
    game_id: int
    rallies: int
    contacts: int


def write_video(conn: sqlite3.Connection, manifest: Manifest, entry: VideoEntry, fps: float,
                rallies: list[dict], contacts: list[dict],
                calibration: dict | None = None) -> WriteSummary:
    """rallies / contacts: the rallies.json / contacts.json lists from find_rallies.py."""
    with conn:
        team_id = upsert_roster(conn, manifest)
        delete_game(conn, entry.key)
        game_id = conn.execute(
            "INSERT INTO games (team_id, video_key, opponent, played_on, venue, net_height_m, "
            "net_height_preset, ball, video_path, fps, calibration) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (team_id, entry.key, entry.opponent, entry.played_on, entry.venue,
             entry.net_height_m, entry.net_height_preset, entry.ball, str(entry.path), fps,
             json.dumps(calibration) if calibration else None),
        ).lastrowid
        set_number = min(entry.lineups) if entry.lineups else 1
        lineup = entry.lineups.get(set_number)
        set_id = conn.execute(
            "INSERT INTO sets (game_id, set_number, starting_lineup, start_frame, end_frame) "
            "VALUES (?, ?, ?, ?, ?)",
            (game_id, set_number, json.dumps(lineup) if lineup else None,
             min((r["start_frame"] for r in rallies), default=None),
             max((r["end_frame"] for r in rallies), default=None)),
        ).lastrowid
        n_contacts = 0
        for r in rallies:
            rally_id = conn.execute(
                "INSERT INTO rallies (set_id, rally_number, start_frame, end_frame, start_t, "
                "end_t, segment_confidence, serving_team, reviewed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (set_id, r["index"], r["start_frame"], r["end_frame"], r["start_t"],
                 r["end_t"], r["confidence"], r["serving_team"]),
            ).lastrowid
            for i in r["contacts"]:
                c = contacts[i]
                if c["kind"] != "touch":
                    continue
                x, y, z = c["ball_xyz"] if c["ball_xyz"] is not None else (None, None, None)
                conn.execute(
                    "INSERT INTO contacts (rally_id, t, frame, contact_confidence, team, "
                    "action, action_confidence, ball_x, ball_y, ball_z, reviewed) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (rally_id, c["t"], c["frame"], c["confidence"], c["team"],
                     c.get("action"), c.get("action_confidence"), x, y, z),
                )
                n_contacts += 1
    return WriteSummary(game_id, len(rallies), n_contacts)
