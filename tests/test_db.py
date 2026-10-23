import sqlite3

import pytest

from setter_eyes import db

REQUIRED_TABLES = {
    "games", "sets", "rallies", "players", "tracks", "contacts", "ball_samples", "corrections",
}


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


def test_schema_creates_required_tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert REQUIRED_TABLES <= {r["name"] for r in rows}


def test_schema_is_idempotent(tmp_path):
    path = tmp_path / "events.db"
    db.connect(path).close()
    db.connect(path).close()


def test_contact_round_trip_and_constraints(conn):
    conn.execute("INSERT INTO teams (id, name) VALUES (1, 'Us')")
    conn.execute("INSERT INTO players (id, team_id, jersey_number, name) VALUES (1, 1, 7, 'A')")
    conn.execute("INSERT INTO games (id, team_id, video_key, video_path, fps) "
                 "VALUES (1, 1, 'g1', 'x.mp4', 59.94)")
    conn.execute("INSERT INTO sets (id, game_id, set_number, starting_lineup) "
                 "VALUES (1, 1, 1, '[7,1,2,3,4,5]')")
    conn.execute("INSERT INTO rallies (id, set_id, rally_number, start_frame, end_frame, "
                 "start_t, end_t) VALUES (1, 1, 1, 0, 300, 0.0, 5.0)")
    conn.execute("INSERT INTO contacts (rally_id, t, frame, player_id, action, pass_score) "
                 "VALUES (1, 1.2, 72, 1, 'receive', 3)")
    row = conn.execute("SELECT action, pass_score, reviewed FROM contacts").fetchone()
    assert tuple(row) == ("receive", 3, 0)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO contacts (rally_id, t, frame, pass_score) "
                     "VALUES (1, 2.0, 120, 5)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO contacts (rally_id, t, frame) VALUES (999, 2.0, 120)")
