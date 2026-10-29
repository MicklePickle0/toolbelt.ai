import pytest

from setter_eyes import db
from setter_eyes import rosters as rost
from setter_eyes.db.pipeline import write_video
from setter_eyes.labels.manifest import Manifest, VideoEntry


def test_heights():
    assert rost.parse_height("188") == 188.0
    assert rost.parse_height("188 cm") == 188.0
    assert rost.parse_height("6'2\"") == pytest.approx(187.96, abs=0.05)
    assert rost.parse_height("6'") == pytest.approx(182.9, abs=0.05)
    assert rost.parse_height("") is None
    with pytest.raises(rost.RosterError):
        rost.parse_height("tall")
    with pytest.raises(rost.RosterError):
        rost.parse_height("18")                      # 18 cm: a typo


def test_files_round_trip_and_validation(tmp_path):
    r = rost.Roster("venom-16u", "Venom 16U", "2026-27", None,
                    [rost.RosterPlayer(12, "Setter", 185.0, "S"),
                     rost.RosterPlayer(6, "Libero", None, "L", True)])
    rost.save(tmp_path, r)
    back = rost.load_all(tmp_path)["venom-16u"]
    assert [p.number for p in back.players] == [6, 12] and back.players[0].libero
    assert back.players[1].height_cm == 185.0 and back.season == "2026-27"
    dup = rost.Roster("x", "X", players=[rost.RosterPlayer(1, "A"), rost.RosterPlayer(1, "B")])
    with pytest.raises(rost.RosterError):
        rost.save(tmp_path, dup)
    assert rost.slugify("Venom 16U  (club)") == "venom-16u-club"


def _touch(t, team, action):
    return {"kind": "touch", "t": t, "frame": int(t * 60), "confidence": 0.6, "team": team,
            "ball_xyz": None, "action": action, "action_confidence": 0.5}


def test_assigned_roster_is_the_games_team_and_credits_follow(tmp_path):
    conn = db.connect(":memory:")
    manifest = Manifest(tmp_path / "manifest.yaml", "Our Team", [], [])
    entry = VideoEntry("v1", tmp_path / "v1.mp4", venue="Gym", ball="mikasa")
    rally = {"index": 1, "start_frame": 600, "end_frame": 900, "start_t": 10.0, "end_t": 15.0,
             "confidence": 0.5, "serving_team": "us", "contacts": [0]}
    rost.save(tmp_path, rost.Roster("club", "Club 16U", players=[
        rost.RosterPlayer(7, "Seven", 190.0, "OH")]))
    rost.save_assignments(tmp_path, {"v1": "club"})
    write_video(conn, manifest, entry, 60.0, [rally], [_touch(10.0, "us", "serve")],
                rosters_dir=tmp_path)
    team = conn.execute("SELECT t.name, t.roster_slug FROM games g JOIN teams t "
                        "ON g.team_id = t.id WHERE g.video_key = 'v1'").fetchone()
    assert tuple(team) == ("Club 16U", "club")
    p = conn.execute("SELECT id, height_cm, position FROM players WHERE jersey_number = 7"
                     ).fetchone()
    assert (p["height_cm"], p["position"]) == (190.0, "OH")
    # credit the serve to #7, then move the video to another roster that also has a #7
    conn.execute("UPDATE contacts SET player_id = ?", (p["id"],))
    rost.save(tmp_path, rost.Roster("hs", "High School", players=[
        rost.RosterPlayer(7, "Other Seven")]))
    rost.save_assignments(tmp_path, {"v1": "hs"})
    rost.sync_db(conn, tmp_path)
    name = conn.execute("SELECT p.name FROM contacts c JOIN players p ON c.player_id = p.id"
                        ).fetchone()[0]
    assert name == "Other Seven"
    # editing a roster keeps player ids; a credited player taken off stays in the DB
    hs7 = conn.execute("SELECT c.player_id FROM contacts c").fetchone()[0]
    rost.save(tmp_path, rost.Roster("hs", "High School", players=[]))
    rost.sync_db(conn, tmp_path)
    assert conn.execute("SELECT 1 FROM players WHERE id = ?", (hs7,)).fetchone()
    with pytest.raises(rost.RosterError):
        rost.delete(tmp_path, "hs")                  # still applied to v1
    rost.delete(tmp_path, "club")
    assert "club" not in rost.load_all(tmp_path)


def test_old_databases_get_the_roster_columns(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    conn.execute("ALTER TABLE teams DROP COLUMN season")      # make it look like v5
    conn.execute("DROP INDEX idx_teams_roster")
    conn.execute("ALTER TABLE teams DROP COLUMN roster_slug")
    conn.execute("ALTER TABLE players DROP COLUMN height_cm")
    conn.execute("ALTER TABLE players DROP COLUMN position")
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    conn.close()
    conn = db.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(players)")}
    assert {"height_cm", "position"} <= cols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
