"""Write pipeline output (rallies, touches, actions) into the event DB.

The pipeline DB (paths.db_path) has the same schema as the labels DB, so stats queries
and evaluation work on either. Everything written here is automated: rows are
unreviewed and carry their confidences. point_winner is inferred from who serves next
(rally/winner.py, point_winner_source = 'next_serve') and the outcome follows from it; the
review UI's key press (source 'user') overrides both.

Re-writing a video replaces its UNREVIEWED rallies only. Rallies saved in the review UI
(reviewed = 1) are human work: they and their touches are kept as they are (bar a pass
score or player the reviewer hasn't given, filled in as a guess below confidence 1), and pipeline
rallies overlapping one are dropped (also for a rally reviewed as `not_a_rally`, so a
false rally the reviewer dismissed doesn't come back). Rally numbers then follow time order again.

A video is one set (set 1 unless the manifest's lineups say otherwise). With a lineup, each
rally gets its rotation and our serves their server (db/rotation.py).
"""

from __future__ import annotations

import bisect
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
    kept_reviewed: int      # reviewed rallies left untouched (bar guesses, below)
    guesses: int = 0        # pass scores / players guessed on reviewed touches without one


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

        players = {r[0]: r[1] for r in conn.execute(
            "SELECT jersey_number, id FROM players WHERE team_id = ?", (team_id,))}
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
                player = players.get(c.get("jersey")) if c["team"] == "us" else None
                conn.execute(
                    "INSERT INTO contacts (rally_id, t, frame, contact_confidence, team, "
                    "action, action_confidence, ball_x, ball_y, ball_z, pass_score, "
                    "pass_confidence, player_id, player_confidence, reviewed) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (rally_id, c["t"], c["frame"], c["confidence"], c["team"],
                     c.get("action"), c.get("action_confidence"), x, y, z,
                     c.get("pass_score"), c.get("pass_confidence"), player,
                     c.get("player_confidence") if player else None),
                )
                n_contacts += 1
        guesses = _guess_on_reviewed(conn, set_id, contacts, players)
        renumber(conn, set_id)
    apply_rotation(conn, set_id)
    return WriteSummary(game_id, n_rallies, n_contacts, len(kept), guesses)


def _guess_on_reviewed(conn, set_id: int, contacts: list[dict], players: dict[int, int],
                       max_dt_s: float = 0.15) -> int:
    """Pass scores and players the reviewer hasn't given yet, on touches of reviewed
    rallies, from the pipeline touch of the same team within max_dt_s: guesses with the
    pipeline's confidence (< 1), shown as such in review; never over a value already
    there. Returns how many were filled."""
    ours = [c for c in contacts if c["kind"] == "touch" and c["team"] == "us"
            and (c.get("pass_score") is not None or c.get("jersey") is not None)]
    if not ours:
        return 0
    t = sorted(ours, key=lambda c: c["t"])
    times = [c["t"] for c in t]
    filled = 0
    rows = conn.execute(
        "SELECT c.id, c.t, c.action, c.pass_score, c.player_id FROM contacts c "
        "JOIN rallies r ON c.rally_id = r.id WHERE r.set_id = ? AND r.reviewed = 1 "
        "AND c.team = 'us' AND (c.pass_score IS NULL OR c.player_id IS NULL)",
        (set_id,)).fetchall()
    for cid, ct, action, pass_score, player_id in rows:
        k = bisect.bisect_left(times, ct)
        near = [t[j] for j in (k - 1, k) if 0 <= j < len(t) and abs(t[j]["t"] - ct) <= max_dt_s]
        if not near:
            continue
        p = min(near, key=lambda c: abs(c["t"] - ct))
        if pass_score is None and p.get("pass_score") is not None and \
                action in ("receive", "overpass"):
            conn.execute("UPDATE contacts SET pass_score = ?, pass_confidence = ? WHERE id = ?",
                         (p["pass_score"], min(p["pass_confidence"] or 0.0, 0.99), cid))
            filled += 1
        if player_id is None and players.get(p.get("jersey")) is not None:
            conn.execute("UPDATE contacts SET player_id = ?, player_confidence = ? WHERE id = ?",
                         (players[p["jersey"]], min(p["player_confidence"] or 0.0, 0.99), cid))
            filled += 1
    return filled
