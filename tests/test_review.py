from setter_eyes import db
from setter_eyes.db.pipeline import write_video
from setter_eyes.labels.manifest import Manifest, VideoEntry
from setter_eyes.ui import review as rv


def _db(tmp_path):
    conn = db.connect(":memory:")
    manifest = Manifest(tmp_path / "manifest.yaml", "Venom", [], [])
    entry = VideoEntry("v1", tmp_path / "v1.mp4", venue="Humber", ball="mikasa")

    def touch(t, team, action):
        return {"kind": "touch", "t": t, "frame": int(t * 60), "confidence": 0.6, "team": team,
                "ball_xyz": None, "action": action, "action_confidence": 0.5}

    contacts = [touch(10.0, "them", "serve"), touch(11.0, "us", "receive"),
                touch(12.0, "us", "set"), touch(13.0, "us", "dig"), touch(16.0, "us", "dig")]
    rally = {"index": 1, "start_frame": 600, "end_frame": 1000, "start_t": 10.0,
             "end_t": 16.5, "confidence": 0.4, "serving_team": "them",
             "contacts": [0, 1, 2, 3, 4]}
    write_video(conn, manifest, entry, 60.0, [rally], contacts)
    g = rv.games(conn)[0]
    return conn, g, rv.rallies(conn, g["set_id"])[0]


def test_save_review_fixes_touches_and_logs_corrections(tmp_path):
    conn, g, rally = _db(tmp_path)
    t = rv.touches(conn, rally["id"])
    edits = [rv.TouchEdit(x["id"], x["t"], x["team"], x["action"]) for x in t]
    edits[3].action = "attack"                         # the "dig" was our attack
    edits.append(rv.TouchEdit(None, 12.6, "us", "set"))  # a missed touch, added
    assert rv.suggest_outcome(edits, "us") == "other"  # a caught ball at 16 s looks last...
    end_t = 14.0                                       # ...until the rally ends at 14 s
    assert rv.suggest_outcome(edits, "us", end_t) == "kill"
    rv.save_review(conn, rally["id"], 60.0, "us", "kill", "them", end_t, edits, True)
    r = rv.rallies(conn, g["set_id"])[0]
    assert (r["point_winner"], r["outcome"], r["reviewed"], r["end_t"]) == \
        ("us", "kill", 1, 14.0)
    after = rv.touches(conn, rally["id"])
    assert [x["action"] for x in after] == ["serve", "receive", "set", "set", "attack"]
    assert all(x["reviewed"] == 1 for x in after)
    fields = {(c["table_name"], c["field"]) for c in conn.execute("SELECT * FROM corrections")}
    assert {("contacts", "action"), ("contacts", "insert"), ("contacts", "delete"),
            ("rallies", "point_winner"), ("rallies", "end_t")} <= fields
    assert rv.score(rv.rallies(conn, g["set_id"])) == (1, 0)
    assert rv.review_queue(rv.rallies(conn, g["set_id"])) == []


def test_a_replay_scores_nothing(tmp_path):
    conn, g, rally = _db(tmp_path)
    edits = [rv.TouchEdit(x["id"], x["t"], x["team"], x["action"])
             for x in rv.touches(conn, rally["id"])]
    assert rv.suggest_outcome(edits, rv.REPLAY) == rv.REPLAY
    rv.save_review(conn, rally["id"], 60.0, rv.REPLAY, None, "them", 16.5, edits, False)
    r = rv.rallies(conn, g["set_id"])[0]
    assert (r["point_winner"], r["outcome"]) == (None, "replay")
    assert rv.score(rv.rallies(conn, g["set_id"])) == (0, 0)


def test_a_false_rally_is_dismissed_and_stays_dismissed(tmp_path):
    conn, g, rally = _db(tmp_path)
    rv.save_review(conn, rally["id"], 60.0, rv.NOT_A_RALLY, None, None, 16.5, [], False)
    r = rv.rallies(conn, g["set_id"])[0]
    assert (r["point_winner"], r["outcome"], r["reviewed"]) == (None, rv.NOT_A_RALLY, 1)
    assert rv.score(rv.rallies(conn, g["set_id"])) == (0, 0)
