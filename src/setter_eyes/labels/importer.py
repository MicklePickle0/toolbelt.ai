"""Write validated labels into an event DB (normally the ground-truth labels DB).

Labels are ground truth: rows are marked reviewed with confidence 1.0.
Re-importing a video replaces everything previously imported for it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from setter_eyes.db.rotation import apply_rotation
from setter_eyes.ingest.video import VideoInfo
from setter_eyes.labels.boxes import BoxLabels
from setter_eyes.labels.events import RallyLabel
from setter_eyes.labels.manifest import Manifest, VideoEntry


@dataclass(frozen=True)
class ImportSummary:
    game_id: int
    sets: int
    rallies: int
    contacts: int
    ball_samples: int


def upsert_roster(conn: sqlite3.Connection, manifest: Manifest) -> int:
    """Ensure our team and its roster exist; returns the team id."""
    conn.execute("INSERT OR IGNORE INTO teams (name) VALUES (?)", (manifest.team,))
    team_id = conn.execute("SELECT id FROM teams WHERE name = ?", (manifest.team,)).fetchone()[0]
    for p in manifest.roster:
        conn.execute(
            "INSERT INTO players (team_id, jersey_number, name, is_libero) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (team_id, jersey_number) DO UPDATE SET name = excluded.name, "
            "is_libero = excluded.is_libero",
            (team_id, p.number, p.name, int(p.libero)),
        )
    return team_id


def delete_game(conn: sqlite3.Connection, video_key: str) -> None:
    row = conn.execute("SELECT id FROM games WHERE video_key = ?", (video_key,)).fetchone()
    if row is None:
        return
    game_id = row[0]
    rally_ids = "SELECT r.id FROM rallies r JOIN sets s ON r.set_id = s.id WHERE s.game_id = ?"
    conn.execute(f"DELETE FROM contacts WHERE rally_id IN ({rally_ids})", (game_id,))
    conn.execute("DELETE FROM ball_samples WHERE game_id = ?", (game_id,))
    conn.execute(f"DELETE FROM rallies WHERE id IN ({rally_ids})", (game_id,))
    conn.execute("DELETE FROM sets WHERE game_id = ?", (game_id,))
    conn.execute("DELETE FROM tracks WHERE game_id = ?", (game_id,))
    conn.execute("DELETE FROM games WHERE id = ?", (game_id,))


def import_video(conn: sqlite3.Connection, manifest: Manifest, entry: VideoEntry,
                 info: VideoInfo, rallies: list[RallyLabel],
                 boxes: BoxLabels | None = None) -> ImportSummary:
    """Replace this video's labels in the DB in one transaction."""
    with conn:
        team_id = upsert_roster(conn, manifest)
        # Jerseys labelled but not on the roster become players named by number, so
        # per-player stats work before the roster is entered (it fills in the names).
        for n in sorted({c.jersey for r in rallies for c in r.contacts
                         if c.team == "us" and c.jersey is not None}):
            conn.execute("INSERT OR IGNORE INTO players (team_id, jersey_number, name) "
                         "VALUES (?, ?, ?)", (team_id, n, f"#{n}"))
        players = {
            r["jersey_number"]: r["id"]
            for r in conn.execute("SELECT id, jersey_number FROM players WHERE team_id = ?",
                                  (team_id,))
        }
        delete_game(conn, entry.key)
        game_id = conn.execute(
            "INSERT INTO games (team_id, video_key, opponent, played_on, venue, net_height_m, "
            "net_height_preset, ball, video_path, fps, frame_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (team_id, entry.key, entry.opponent, entry.played_on, entry.venue,
             entry.net_height_m, entry.net_height_preset, entry.ball, str(entry.path), info.fps,
             info.frame_count),
        ).lastrowid

        set_ids: dict[int, int] = {}
        set_numbers = sorted({r.set_number for r in rallies} | set(entry.lineups))
        for n in set_numbers:
            in_set = [r for r in rallies if r.set_number == n]
            lineup = entry.lineups.get(n)
            set_ids[n] = conn.execute(
                "INSERT INTO sets (game_id, set_number, starting_lineup, start_frame, end_frame) "
                "VALUES (?, ?, ?, ?, ?)",
                (game_id, n, json.dumps(lineup) if lineup else None,
                 min((r.start_frame for r in in_set), default=None),
                 max((r.end_frame for r in in_set), default=None)),
            ).lastrowid

        n_contacts = 0
        rally_spans: list[tuple[int, int, int]] = []
        for r in rallies:
            rally_id = conn.execute(
                "INSERT INTO rallies (set_id, rally_number, start_frame, end_frame, start_t, "
                "end_t, segment_confidence, serving_team, point_winner, point_winner_source, "
                "outcome, outcome_confidence, reviewed) "
                "VALUES (?, ?, ?, ?, ?, ?, 1.0, ?, ?, 'user', ?, ?, 1)",
                (set_ids[r.set_number], r.rally_number, r.start_frame, r.end_frame,
                 r.start_frame / info.fps, r.end_frame / info.fps, r.serving_team,
                 r.point_winner, r.outcome, 1.0 if r.outcome else None),
            ).lastrowid
            rally_spans.append((r.start_frame, r.end_frame, rally_id))
            for c in r.contacts:
                conn.execute(
                    "INSERT INTO contacts (rally_id, t, frame, contact_confidence, team, "
                    "player_id, player_confidence, action, action_confidence, pass_score, "
                    "pass_confidence, reviewed) VALUES (?, ?, ?, 1.0, ?, ?, ?, ?, ?, ?, ?, 1)",
                    (rally_id, c.frame / info.fps, c.frame, c.team,
                     players.get(c.jersey) if c.jersey is not None else None,
                     1.0 if c.jersey is not None else None,
                     c.action, 1.0 if c.action else None,
                     c.pass_score, 1.0 if c.pass_score is not None else None),
                )
                n_contacts += 1

        n_ball = 0
        if boxes is not None:
            for frame, frame_boxes in sorted(boxes.frames.items()):
                rally_id = next((rid for s, e, rid in rally_spans if s <= frame <= e), None)
                for b in frame_boxes:
                    if b.cls != "ball":
                        continue
                    u, v = b.centre
                    conn.execute(
                        "INSERT INTO ball_samples (game_id, rally_id, frame, t, u, v, "
                        "confidence, source) VALUES (?, ?, ?, ?, ?, ?, 1.0, 'label')",
                        (game_id, rally_id, frame, frame / info.fps, u, v),
                    )
                    n_ball += 1

    for set_id in set_ids.values():
        apply_rotation(conn, set_id)
    return ImportSummary(game_id, len(set_ids), len(rallies), n_contacts, n_ball)
