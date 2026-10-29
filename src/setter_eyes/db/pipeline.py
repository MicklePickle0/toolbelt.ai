"""Write pipeline output (rallies, touches, actions) into the event DB.

The pipeline DB (paths.db_path) has the same schema as the labels DB, so stats queries
and evaluation work on either. Everything written here is automated: rows are
unreviewed and carry their confidences. point_winner is inferred from who serves next
(rally/winner.py, point_winner_source = 'next_serve') and the outcome follows from it; the
review UI's key press (source 'user') overrides both.

Re-writing a video replaces its UNREVIEWED rallies only. Rallies saved in the review UI
(reviewed = 1) are human work: they and their touches are kept as they are, and pipeline
rallies overlapping one are dropped (also for a rally reviewed as `not_a_rally`, so a
false rally the reviewer dismissed doesn't come back). Rally numbers then follow time order again.

A video is one set (set 1 unless the manifest's lineups say otherwise). With a lineup, each
rally gets its rotation and our serves their server (db/rotation.py).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from setter_eyes import rosters as rost
from setter_eyes.db.rotation import apply_rotation
from setter_eyes.labels.importer import upsert_roster
from setter_eyes.labels.manifest import Manifest, VideoEntry


@dataclass(frozen=True)
class WriteSummary:
    game_id: int
    rallies: int            # pipeline rallies written
    contacts: int
    kept_reviewed: int      # reviewed rallies left untouched


def _game(conn, team_id: int, entry: VideoEntry, fps: float, calibration) -> int:
    values = (team_id, entry.opponent, entry.played_on, entry.venue, entry.net_height_m,
              entry.net_height_preset, entry.ball, str(entry.path), fps,
              json.dumps(calibration) if calibration else None)
    row = conn.execute("SELECT id FROM games WHERE video_key = ?", (entry.key,)).fetchone()
    if row is not None:
        conn.execute("UPDATE games SET team_id = ?, opponent = ?, played_on = ?, venue = ?, "
                     "net_height_m = ?, net_height_preset = ?, ball = ?, video_path = ?, "
                     "fps = ?, calibration = ? WHERE id = ?", (*values, row[0]))
        return row[0]
    return conn.execute(
        "INSERT INTO games (team_id, opponent, played_on, venue, net_height_m, "
        "net_height_preset, ball, video_path, fps, calibration, video_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (*values, entry.key)).lastrowid


def renumber(conn: sqlite3.Connection, set_id: int) -> None:
    """Rally numbers 1..n in time order (two steps: (set_id, rally_number) is unique)."""
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM rallies WHERE set_id = ? ORDER BY start_t", (set_id,))]
    for n, rid in enumerate(ids, start=1):
        conn.execute("UPDATE rallies SET rally_number = ? WHERE id = ?", (-n, rid))
    conn.execute("UPDATE rallies SET rally_number = -rally_number WHERE set_id = ?", (set_id,))


def write_video(conn: sqlite3.Connection, manifest: Manifest, entry: VideoEntry, fps: float,
                rallies: list[dict], contacts: list[dict],
                calibration: dict | None = None,
                rosters_dir: Path | None = None) -> WriteSummary:
    """rallies / contacts: the rallies.json / contacts.json lists from find_rallies.py.
    rosters_dir: data/rosters; the video's assigned roster (rosters.py), if any, is its team."""
    team_id = rost.team_for_video(conn, rosters_dir, entry.key)
    with conn:
        manifest_team = upsert_roster(conn, manifest)
        team_id = team_id if team_id is not None else manifest_team
        game_id = _game(conn, team_id, entry, fps, calibration)
        set_number = min(entry.lineups) if entry.lineups else 1
        lineup = entry.lineups.get(set_number)
        row = conn.execute("SELECT id FROM sets WHERE game_id = ? AND set_number = ?",
                           (game_id, set_number)).fetchone()
        set_id = row[0] if row else conn.execute(
            "INSERT INTO sets (game_id, set_number) VALUES (?, ?)",
            (game_id, set_number)).lastrowid
        conn.execute("UPDATE sets SET starting_lineup = ?, start_frame = ?, end_frame = ? "
                     "WHERE id = ?",
                     (json.dumps(lineup) if lineup else None,
                      min((r["start_frame"] for r in rallies), default=None),
                      max((r["end_frame"] for r in rallies), default=None), set_id))

        # Drop the previous automated rallies; keep the reviewed ones.
        old = "SELECT id FROM rallies WHERE set_id = ? AND reviewed = 0"
        conn.execute(f"DELETE FROM contacts WHERE rally_id IN ({old})", (set_id,))
        conn.execute(f"DELETE FROM rallies WHERE id IN ({old})", (set_id,))
        kept = [(r[0], r[1]) for r in conn.execute(
            "SELECT start_t, end_t FROM rallies WHERE set_id = ?", (set_id,))]

        n_rallies = n_contacts = 0
        for r in rallies:
            if any(min(r["end_t"], b) > max(r["start_t"], a) for a, b in kept):
                continue                          # a reviewed rally already covers it
            n_rallies += 1
            rally_id = conn.execute(
                "INSERT INTO rallies (set_id, rally_number, start_frame, end_frame, start_t, "
                "end_t, segment_confidence, serving_team, point_winner, point_winner_source, "
                "score_us, score_them, outcome, outcome_confidence, reviewed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (set_id, 100_000 + n_rallies, r["start_frame"], r["end_frame"], r["start_t"],
                 r["end_t"], r["confidence"], r["serving_team"], r.get("point_winner"),
                 "next_serve" if r.get("point_winner") else None, r.get("score_us"),
                 r.get("score_them"), r.get("outcome"), r.get("outcome_confidence")),
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
        renumber(conn, set_id)
    apply_rotation(conn, set_id)
    return WriteSummary(game_id, n_rallies, n_contacts, len(kept))
