import json
import sqlite3

from setter_eyes import config, db
from setter_eyes.db.rotation import apply_rotation
from setter_eyes.identity.rotation import follow_rotation, suggest_lineup

CFG = config.load("pipeline")
LINEUP = [18, 12, 14, 1, 13, 5]


def test_we_rotate_when_we_win_the_serve_back():
    states = follow_rotation(LINEUP, ["us", "us", "them", "them", "us", "them", "us"], CFG)
    assert [s.rotation for s in states] == [1, 1, 1, 1, 2, 2, 3]
    assert [s.server for s in states] == [18, 18, None, None, 12, None, 14]
    assert states[4].positions == {1: 12, 2: 14, 3: 1, 4: 13, 5: 5, 6: 18}


def test_receiving_first_the_second_slot_serves_first():
    states = follow_rotation(LINEUP, ["them", "us"], CFG)
    assert states[1].server == 12


def test_uncertain_side_outs_lower_confidence_and_a_known_server_restores_it():
    teams = ["us", "them", "us", "them", "us"]
    states = follow_rotation(LINEUP, teams, CFG, confidences=[1, 1, 0.5, 1, 0.5])
    assert [s.confidence for s in states] == [1.0, 1.0, 0.5, 0.5, 0.25]
    states = follow_rotation(LINEUP, teams, CFG, confidences=[1, 1, 0.5, 1, 0.5],
                             servers=[None, None, 12, None, None])
    assert states[4].confidence == 0.5


def test_a_server_not_on_court_is_a_substitution_into_that_slot():
    teams = ["us", "them", "us", "them", "us"]
    states = follow_rotation(LINEUP, teams, CFG, servers=[None, None, 10, None, None])
    assert states[2].flags == ["substitution"] and states[2].server == 10
    assert 12 not in states[4].positions.values() and states[4].positions[6] == 10


def test_a_server_from_another_slot_resyncs_the_rotation():
    # A missed side-out: 14 serves where 12 was expected.
    states = follow_rotation(LINEUP, ["us", "them", "us", "them", "us"], CFG,
                             servers=[None, None, 14, None, None])
    assert states[2].flags == ["resync"] and states[2].rotation == 3
    assert states[4].server == 1


def test_unknown_serving_team_keeps_the_rotation_but_lowers_confidence():
    states = follow_rotation(LINEUP, ["them", None, "us"], CFG)
    assert states[1].flags == ["unknown_serve"] and states[2].server == 12
    assert states[2].confidence == CFG["identity"]["rotation"]["unknown_serve_factor"]


def test_suggest_lineup_from_labelled_servers():
    teams = ["them", "us", "them", "us", "them", "us", "us", "them", "us"]
    servers = [None, 12, None, 14, None, 1, 1, None, 10]
    slots, notes = suggest_lineup(teams, servers)
    assert slots == [None, 12, 14, 1, 10, None]
    assert notes == []
    slots, notes = suggest_lineup(["us", "them", "us"], [18, None, 18])
    assert slots == [18, None, None, None, None, None] and "missed or false" in notes[0]


def _set_with_rallies(conn, lineup, rallies):
    """rallies: (serving_team, server jersey or None, reviewed)."""
    conn.execute("INSERT INTO teams (id, name) VALUES (1, 'Us')")
    for n in (18, 12):
        conn.execute("INSERT INTO players (team_id, jersey_number, name) VALUES (1, ?, ?)",
                     (n, f"#{n}"))
    conn.execute("INSERT INTO games (id, team_id, video_key, venue, net_height_m, ball, "
                 "video_path, fps) VALUES (1, 1, 'g1', 'Gym', 2.43, 'mikasa_v200w_yellow', "
                 "'x.mp4', 30)")
    conn.execute("INSERT INTO sets (id, game_id, set_number, starting_lineup) VALUES (1, 1, 1, ?)",
                 (json.dumps(lineup) if lineup else None,))
    for i, (team, jersey, reviewed) in enumerate(rallies, start=1):
        rid = conn.execute(
            "INSERT INTO rallies (set_id, rally_number, start_frame, end_frame, start_t, end_t, "
            "segment_confidence, serving_team, reviewed) VALUES (1, ?, 0, 1, ?, ?, 0.5, ?, ?)",
            (i, 10.0 * i, 10.0 * i + 5, team, reviewed)).lastrowid
        pid = conn.execute("SELECT id FROM players WHERE jersey_number = ?",
                           (jersey,)).fetchone() if jersey else None
        conn.execute("INSERT INTO contacts (rally_id, t, frame, team, action, player_id, "
                     "reviewed) VALUES (?, ?, 0, ?, 'serve', ?, ?)",
                     (rid, 10.0 * i, team, pid[0] if pid else None, reviewed))


def test_apply_rotation_names_our_servers_with_confidence():
    conn = db.connect(":memory:")
    _set_with_rallies(conn, LINEUP, [("us", None, 0), ("them", None, 0), ("us", None, 0),
                                     ("them", None, 0), ("us", None, 0)])
    summary = apply_rotation(conn, 1, CFG)
    assert summary.rallies == 5 and summary.servers_set == 3
    rows = conn.execute(
        "SELECT r.rotation, p.jersey_number, c.player_confidence FROM rallies r "
        "JOIN contacts c ON c.rally_id = r.id LEFT JOIN players p ON c.player_id = p.id "
        "WHERE c.team = 'us' ORDER BY r.start_t").fetchall()
    assert [tuple(r) for r in rows] == [(1, 18, 1.0), (2, 12, 0.5), (3, 14, 0.25)]


def test_apply_rotation_keeps_reviewed_servers_and_anchors_on_them():
    conn = db.connect(":memory:")
    _set_with_rallies(conn, LINEUP, [("us", None, 0), ("them", None, 0), ("us", 18, 1)])
    summary = apply_rotation(conn, 1, CFG)
    assert summary.anchors == 1
    rows = conn.execute("SELECT r.rotation, c.player_id, c.player_confidence FROM rallies r "
                        "JOIN contacts c ON c.rally_id = r.id ORDER BY r.start_t").fetchall()
    assert rows[2]["rotation"] == 1                 # resynced to 18 serving
    assert rows[2]["player_confidence"] is None     # the reviewed row is left as it was


def test_apply_rotation_without_a_lineup_does_nothing():
    conn = db.connect(":memory:")
    _set_with_rallies(conn, None, [("us", None, 0)])
    assert apply_rotation(conn, 1, CFG) is None
    assert conn.execute("SELECT rotation FROM rallies").fetchone()[0] is None


def test_v4_db_gains_player_confidence(tmp_path):
    path = tmp_path / "v4.db"
    conn = db.connect(path)
    conn.execute("ALTER TABLE contacts DROP COLUMN player_confidence")
    conn.execute("PRAGMA user_version = 4")
    conn.close()
    conn = db.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(contacts)")}
    assert "player_confidence" in cols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert isinstance(conn, sqlite3.Connection)
