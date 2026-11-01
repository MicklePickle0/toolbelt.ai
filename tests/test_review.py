import pytest

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


def test_a_split_point_is_merged_and_its_start_moved(tmp_path):
    conn = db.connect(":memory:")
    manifest = Manifest(tmp_path / "manifest.yaml", "Venom", [], [])
    entry = VideoEntry("v1", tmp_path / "v1.mp4", venue="Humber", ball="mikasa")

    def touch(t, team, action):
        return {"kind": "touch", "t": t, "frame": int(t * 60), "confidence": 0.6, "team": team,
                "ball_xyz": None, "action": action, "action_confidence": 0.5}

    contacts = [touch(11.0, "us", "receive"), touch(12.0, "us", "set"),
                touch(14.5, "them", "dig"), touch(15.5, "them", "set")]

    def rally(i, t0, t1, members):
        return {"index": i, "start_frame": int(t0 * 60), "end_frame": int(t1 * 60),
                "start_t": t0, "end_t": t1, "confidence": 0.4, "serving_team": "them",
                "contacts": members}

    write_video(conn, manifest, entry, 60.0,
                [rally(1, 10.9, 12.5, [0, 1]), rally(2, 14.4, 16.0, [2, 3])], contacts)
    g = rv.games(conn)[0]
    first, second = rv.rallies(conn, g["set_id"])
    assert rv.neighbours(rv.rallies(conn, g["set_id"]), first["id"]) == (None, second)
    rv.merge_rallies(conn, first["id"], second["id"])
    (merged,) = rv.rallies(conn, g["set_id"])
    assert (merged["start_t"], merged["end_t"]) == (10.9, 16.0)
    assert len(rv.touches(conn, merged["id"])) == 4
    # The serve wasn't detected: move the start back to it and add the serve.
    edits = [rv.TouchEdit(x["id"], x["t"], x["team"], x["action"])
             for x in rv.touches(conn, merged["id"])]
    edits.append(rv.TouchEdit(None, 10.0, "them", "serve"))
    rv.save_review(conn, merged["id"], 60.0, "them", None, "them", 16.0, edits, True,
                   start_t=10.0)
    r = rv.rallies(conn, g["set_id"])[0]
    assert r["start_t"] == 10.0
    assert [x["action"] for x in rv.touches(conn, r["id"])][0] == "serve"
    fields = {c["field"] for c in conn.execute("SELECT field FROM corrections")}
    assert {"merge", "start_t", "insert"} <= fields


def test_reviewed_rallies_become_training_labels(tmp_path):
    from setter_eyes.db.read import reviewed_for_training

    conn, g, rally = _db(tmp_path)
    edits = [rv.TouchEdit(x["id"], x["t"], x["team"], x["action"])
             for x in rv.touches(conn, rally["id"])]
    assert reviewed_for_training(conn, "v1") == []                    # nothing reviewed yet
    rv.save_review(conn, rally["id"], 60.0, "us", "kill", "them", 16.5, edits, False)
    assert reviewed_for_training(conn, "v1") == []                    # touches not confirmed
    rv.save_review(conn, rally["id"], 60.0, "us", "kill", "them", 16.5, edits, True)
    (r,) = reviewed_for_training(conn, "v1")
    assert len(r.contacts) == 5 and r.point_winner == "us"
    rv.save_review(conn, rally["id"], 60.0, rv.NOT_A_RALLY, None, None, 16.5, edits, False)
    (r,) = reviewed_for_training(conn, "v1")
    assert r.contacts == []                                           # all false touches


def test_split_two_points_counted_as_one(tmp_path):
    conn = db.connect(":memory:")
    manifest = Manifest(tmp_path / "manifest.yaml", "Venom", [], [])
    entry = VideoEntry("v1", tmp_path / "v1.mp4", venue="Humber", ball="mikasa")

    def touch(t, team, action):
        return {"kind": "touch", "t": t, "frame": int(t * 60), "confidence": 0.6, "team": team,
                "ball_xyz": None, "action": action, "action_confidence": 0.5}

    # point 1: their serve, our dig... point 2 from 20 s: our serve, their dig
    contacts = [touch(10.0, "them", "serve"), touch(11.0, "us", "receive"),
                touch(12.0, "us", "attack"), touch(20.0, "us", "serve"),
                touch(21.0, "them", "receive")]
    rally = {"index": 1, "start_frame": 600, "end_frame": 1500, "start_t": 10.0,
             "end_t": 25.0, "confidence": 0.4, "serving_team": "them",
             "contacts": [0, 1, 2, 3, 4], "point_winner": "them", "outcome": "other"}
    write_video(conn, manifest, entry, 60.0, [rally], contacts)
    g = rv.games(conn)[0]
    rid = rv.rallies(conn, g["set_id"])[0]["id"]
    with pytest.raises(ValueError):
        rv.split_rally(conn, rid, 30.0)                  # outside the rally
    new = rv.split_rally(conn, rid, 19.99)               # 10 ms after the serve still works
    rows = rv.rallies(conn, g["set_id"])
    assert [r["id"] for r in rows] == [rid, new] and [r["rally_number"] for r in rows] == [1, 2]
    first, second = rows
    assert (first["end_t"], second["start_t"], second["end_t"]) == (19.99, 19.99, 25.0)
    assert [x["action"] for x in rv.touches(conn, rid)] == ["serve", "receive", "attack"]
    assert [x["action"] for x in rv.touches(conn, new)] == ["serve", "receive"]
    # we serve the second point, so we won the first: our attack was a kill
    assert (first["point_winner"], first["outcome"]) == ("us", "kill")
    assert (second["serving_team"], second["point_winner"]) == ("us", "them")
    assert not first["reviewed"] and not second["reviewed"]
    assert conn.execute("SELECT field FROM corrections").fetchone()[0] == "split"


def test_review_saves_pass_score_and_player(tmp_path):
    from setter_eyes.labels.manifest import RosterEntry
    conn = db.connect(":memory:")
    manifest = Manifest(tmp_path / "manifest.yaml", "Venom", [RosterEntry(7, "Sam")], [])
    entry = VideoEntry("v1", tmp_path / "v1.mp4", venue="Humber", ball="mikasa")
    contacts = [{"kind": "touch", "t": t, "frame": int(t * 60), "confidence": 0.6, "team": team,
                 "ball_xyz": None, "action": a, "action_confidence": 0.5, "pass_score": p,
                 "pass_confidence": None if p is None else 0.4}
                for t, team, a, p in ((10.0, "them", "serve", None), (11.0, "us", "receive", 2),
                                      (12.0, "us", "set", None))]
    write_video(conn, manifest, entry, 60.0, [{
        "index": 1, "start_frame": 600, "end_frame": 800, "start_t": 10.0, "end_t": 13.0,
        "confidence": 0.4, "serving_team": "them", "contacts": [0, 1, 2]}], contacts)
    rid = rv.rallies(conn, rv.games(conn)[0]["set_id"])[0]["id"]
    t = rv.touches(conn, rid)
    assert (t[1]["pass_score"], t[1]["pass_confidence"]) == (2, 0.4)   # from the pipeline
    edits = [rv.TouchEdit(x["id"], x["t"], x["team"], x["action"]) for x in t]
    edits[1] = rv.TouchEdit(t[1]["id"], 11.0, "us", "receive", pass_score=3, jersey=7,
                            keep_extra=False)
    edits[0] = rv.TouchEdit(t[0]["id"], 10.0, "them", "serve", jersey=7, keep_extra=False)
    rv.save_review(conn, rid, 60.0, "us", "kill", "them", 13.0, edits, True)
    t = rv.touches(conn, rid)
    assert (t[1]["pass_score"], t[1]["pass_confidence"], t[1]["jersey"]) == (3, 1.0, 7)
    assert t[0]["jersey"] is None                        # their touch: no player of ours
    assert t[2]["pass_score"] is None                    # a set has no pass score
    logged = {(c["field"], c["new_value"]) for c in conn.execute("SELECT * FROM corrections")}
    assert {("pass_score", "3"), ("jersey", "7")} <= logged
    # A caller that doesn't send them (merge, Streamlit) keeps them.
    rv.save_review(conn, rid, 60.0, "us", "kill", "them", 13.0,
                   [rv.TouchEdit(x["id"], x["t"], x["team"], x["action"])
                    for x in rv.touches(conn, rid)], True)
    assert rv.touches(conn, rid)[1]["pass_score"] == 3
    bad = [rv.TouchEdit(x["id"], x["t"], x["team"], x["action"], jersey=99, keep_extra=False)
           for x in rv.touches(conn, rid)]
    with pytest.raises(ValueError, match="roster"):
        rv.save_review(conn, rid, 60.0, "us", "kill", "them", 13.0, bad, True)
