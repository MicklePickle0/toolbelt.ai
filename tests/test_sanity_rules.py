"""The owner's volleyball sanity rules (2026-10-28): rally merging and gaps, overpasses,
kills on a ball the other side couldn't control, assists, blocks from the landing point,
and the review tools for points the pipeline missed or joined."""

import pytest

from setter_eyes import config, db
from setter_eyes.classify.outcome import rally_outcome
from setter_eyes.classify.rules import classify_rally
from setter_eyes.contacts.blocks import blocks_from_landing
from setter_eyes.contacts.events import Contact
from setter_eyes.db.pipeline import write_video
from setter_eyes.labels.events import ContactLabel, RallyLabel
from setter_eyes.labels.manifest import Manifest, VideoEntry
from setter_eyes.rally.segment import Rally, flag_gaps, merge_close
from setter_eyes.stats.sheet import stat_rows
from setter_eyes.ui import review as rv

CFG = config.load("pipeline")


def _c(t, team, kind="touch", xyz=None, conf=0.9, action=None):
    return Contact(0, int(t * 60), t, kind, team, -1, xyz, [0.0, 0.0], 3.0, 0.5, conf,
                   action=action, action_confidence=0.8 if action else None)


def _rally(i, t0, t1, members, serve=None, conf=0.7):
    return Rally(i, int(t0 * 60), int(t1 * 60), t0, t1, "us", serve, "floor", members,
                 confidence=conf)


def test_rallies_under_4_s_apart_merge_unless_a_clear_serve_starts_the_second():
    contacts = [_c(10, "us"), _c(12, "them"), _c(14.5, "them", conf=0.5), _c(30, "us"),
                _c(33, "them", conf=0.95)]
    rallies = [_rally(1, 10, 12.5, [0, 1], serve=0), _rally(2, 14.5, 16, [2], serve=2),
               _rally(3, 30, 31, [3], serve=3), _rally(4, 33, 35, [4], serve=4)]
    out = merge_close(rallies, contacts, CFG)
    # 2 joins 1 (2.0 s apart, unsure "serve"); 4 is 2.0 s after 3 but starts with a sure serve
    assert [(r.index, r.start_t, r.end_t) for r in out] == [(1, 10, 16), (2, 30, 31), (3, 33, 35)]
    assert out[0].contacts == [0, 1, 2] and not contacts[2].serve


def test_downtime_piece_before_a_serve_is_dropped_not_merged():
    contacts = [_c(5, "them", conf=0.4), _c(8, "us", conf=0.6)]
    rallies = [_rally(1, 5, 6, [0], serve=None), _rally(2, 8, 12, [1], serve=1)]
    out = merge_close(rallies, contacts, CFG)
    assert [(r.start_t, r.contacts) for r in out] == [(8, [1])]


def test_a_long_gap_lowers_the_next_rallys_confidence():
    rallies = [_rally(1, 0, 5, []), _rally(2, 18, 20, []), _rally(3, 50, 55, [])]
    assert flag_gaps(rallies, CFG) == 1
    assert rallies[1].confidence == 0.7 and rallies[2].confidence < 0.7


def test_a_lofted_single_touch_sent_straight_back_is_an_overpass():
    serve, rec, s, att, dig = (_c(1, "them"), _c(2, "us"), _c(3, "us"), _c(4, "us"),
                               _c(5.5, "them"))
    back = _c(7, "us")
    touches = [serve, rec, s, att, dig, back]
    launches = [None, None, None, None, 6.0, None]    # their dig lofted straight back over
    acts = [a for a, _ in classify_rally(touches, serve, "them", CFG["classify"], launches)]
    assert acts[4] == "overpass" and acts[1] == "receive" and acts[3] == "attack"


def test_a_ball_sent_over_that_the_other_side_cannot_control_is_a_kill():
    us = "us"
    assert rally_outcome([(us, "set", 0.8), (us, "free_ball", 0.7), ("them", "dig", 0.6)],
                         us) == ("kill", 0.7)
    assert rally_outcome([(us, "attack", 0.9), ("them", "dig", 0.6)], us)[0] == "kill"
    # two touches by them before the error: their ball-handling error, not our kill
    assert rally_outcome([(us, "attack", 0.9), ("them", "dig", 0.6), ("them", "set", 0.6)],
                         us)[0] == "ball_handling_error"
    # a failed serve receive stays a reception error (not a kill)
    assert rally_outcome([("us", "serve", 0.9), ("them", "receive", 0.6)],
                         us)[0] == "reception_error"


def test_assists_and_kills_from_a_free_ball():
    def r(n, contacts, winner, outcome):
        return RallyLabel(1, n, 10 * n, 10 * n + 9, winner, outcome,
                          [ContactLabel(10 * n + i, t, a, j, None)
                           for i, (t, a, j) in enumerate(contacts)])
    rallies = [r(1, [("them", "serve", None), ("us", "receive", 7), ("us", "set", 12),
                     ("us", "attack", 7)], "us", "kill"),
               r(2, [("them", "serve", None), ("us", "receive", 7), ("us", "set", 12),
                     ("us", "free_ball", 5), ("them", "dig", None)], "us", "kill")]
    rows = stat_rows(rallies, config.load("pass_scale"))
    assert rows[12]["assists"] == 1 and rows[7]["kills"] == 1 and rows[5]["kills"] == 1
    assert rows["team"]["kills"] == 2


def test_an_attack_landing_on_the_attackers_own_side_was_blocked():
    contacts = [_c(1, "them", action="serve"), _c(2, "us", action="receive"),
                _c(3, "us", action="set"), _c(4, "us", action="attack"),
                _c(4.6, None, kind="floor", xyz=[4.0, 3.0, 0.0])]
    r = _rally(1, 1, 5, [0, 1, 2, 3, 4])
    r.point_winner, r.winner_confidence = "them", 1.0
    net = [_c(1, "us", action="attack"), _c(1.4, None, kind="floor", xyz=[4.0, 0.4, 0.0])]
    r2 = _rally(2, 0.5, 2, [0, 1])
    assert blocks_from_landing([r], contacts, CFG) == 1
    assert [(c.team, c.action) for c in contacts][4] == ("them", "block")
    assert r.contacts == [0, 1, 2, 3, 4, 5] and r.outcome == "block_point"
    assert blocks_from_landing([r2], net, CFG) == 0      # dropped right under the net


def _db(tmp_path):
    conn = db.connect(":memory:")
    manifest = Manifest(tmp_path / "manifest.yaml", "Venom", [], [])
    entry = VideoEntry("v1", tmp_path / "v1.mp4", venue="Gym", ball="mikasa")

    def touch(t, team, action):
        return {"kind": "touch", "t": t, "frame": int(t * 60), "confidence": 0.6, "team": team,
                "ball_xyz": None, "action": action, "action_confidence": 0.5}

    contacts = [touch(100, "them", "serve"), touch(101, "us", "receive"),
                touch(137.4, "us", "serve"), touch(138.4, "them", "receive")]
    rallies = [{"index": 1, "start_frame": 6000, "end_frame": 6420, "start_t": 100.0,
                "end_t": 107.0, "confidence": 0.5, "serving_team": "them", "contacts": [0, 1],
                "point_winner": "us"},
               {"index": 2, "start_frame": 8240, "end_frame": 8430, "start_t": 137.35,
                "end_t": 140.5, "confidence": 0.5, "serving_team": "us", "contacts": [2, 3]}]
    write_video(conn, manifest, entry, 60.0, rallies, contacts)
    return conn, rv.games(conn)[0]["set_id"]


def test_add_a_point_the_pipeline_missed(tmp_path):
    """Pakmen Black set 1, 2026-10-28: their serve into the net at ~117 s was taken for
    downtime between rallies 5 and 6."""
    conn, set_id = _db(tmp_path)
    with pytest.raises(ValueError):
        rv.add_rally(conn, set_id, 105.0, 120.0, "them")     # overlaps the rally before
    new = rv.add_rally(conn, set_id, 117.0, 119.5, "them")
    rows = rv.rallies(conn, set_id)
    assert [r["id"] for r in rows][1] == new and [r["rally_number"] for r in rows] == [1, 2, 3]
    # winners re-inferred from who serves next: they served the missed point, so they won
    # the first rally; we serve rally 6, so we won the missed one
    assert rows[0]["point_winner"] == "them" and rows[1]["point_winner"] == "us"


def test_split_after_widening_the_start(tmp_path):
    conn, set_id = _db(tmp_path)
    r6 = rv.rallies(conn, set_id)[1]["id"]
    rv.set_span(conn, r6, 117.0, 140.5)                       # widened back in review
    rv.split_rally(conn, r6, 137.3)
    rows = rv.rallies(conn, set_id)
    assert [(r["start_t"], r["end_t"]) for r in rows][1:] == [(117.0, 137.3), (137.3, 140.5)]


def test_end_detail_is_saved_with_the_review(tmp_path):
    conn, set_id = _db(tmp_path)
    r = rv.rallies(conn, set_id)[0]
    edits = [rv.TouchEdit(x["id"], x["t"], x["team"], x["action"])
             for x in rv.touches(conn, r["id"])]
    rv.save_review(conn, r["id"], 60.0, "them", "ball_handling_error", "them", 107.0, edits,
                   True, end_detail="net_touch")
    assert conn.execute("SELECT end_detail FROM rallies WHERE id = ?",
                        (r["id"],)).fetchone()[0] == "net_touch"
