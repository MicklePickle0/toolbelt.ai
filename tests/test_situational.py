"""Setting and situational stats (stats/situational.py, roadmap E7/E8)."""

from __future__ import annotations

from setter_eyes import config, db
from setter_eyes.db.pipeline import write_video
from setter_eyes.labels.manifest import Manifest, RosterEntry, VideoEntry
from setter_eyes.stats.situational import attack_zone, court_zone, in_system, situational

SCALE = config.load("pass_scale")


def test_zones_of_our_half():
    assert court_zone(1.0, 1.0) == "4" and court_zone(4.5, 2.0) == "3"
    assert court_zone(8.0, 1.0) == "2" and court_zone(1.0, 6.0) == "5"
    assert court_zone(4.5, 7.0) == "6" and court_zone(8.0, 8.0) == "1"
    assert court_zone(None, 2.0) is None
    # Along the net from the ball, depth from the feet; a ball depth past the net clamps.
    assert attack_zone({"ball_xy": [8.0, -1.0], "player_xy": [7.0, 1.5]}) == "2"
    assert attack_zone({"ball_xy": [4.0, -0.5], "player_xy": None}) == "3"
    assert attack_zone({"ball_xy": None, "player_xy": [4.5, 6.0]}) == "6"


def test_in_system_from_pass_score_else_setter_distance():
    rule = SCALE["in_system"]
    target = SCALE["target"]
    assert in_system({"action": "receive", "pass_score": 3}, None, rule, target) == "in"
    assert in_system({"action": "receive", "pass_score": 1}, [5.5, 1.0], rule, target) == "out"
    assert in_system({"action": "dig", "pass_score": None}, [5.0, 1.5], rule, target) == "in"
    assert in_system({"action": "dig", "pass_score": None}, [2.0, 5.0], rule, target) == "out"
    assert in_system(None, None, rule, target) == "unknown"


def _db(tmp_path):
    conn = db.connect(":memory:")
    manifest = Manifest(tmp_path / "m.yaml", "Venom",
                        [RosterEntry(n, f"P{n}") for n in (6, 10, 17)], [])
    entry = VideoEntry("v1_set1", tmp_path / "v1.mp4", venue="Humber", ball="mikasa")

    def t(time, team, action, jersey=None, pass_score=None, feet=None, pass_xy=None):
        return {"kind": "touch", "t": time, "frame": int(time * 60), "confidence": 0.9,
                "team": team, "ball_xyz": None, "action": action, "action_confidence": 0.9,
                "jersey": jersey, "player_confidence": 0.9 if jersey else None,
                "pass_score": pass_score, "pass_confidence": 0.9 if pass_score else None,
                "player_xy": feet, "pass_xy": pass_xy}

    contacts = [
        # rally 1: good pass by #6 in zone 6, #17 sets from the target, #10 kills from 4
        t(10.0, "them", "serve"), t(11.0, "us", "receive", 6, 4, [4.5, 7.0], [5.4, 1.1]),
        t(12.0, "us", "set", 17, feet=[5.4, 1.1]), t(13.0, "us", "attack", 10, feet=[1.0, 2.0]),
        # rally 2: bad pass by #6 in zone 1, #17 sets off the net, #10 hits it out
        t(40.0, "them", "serve"), t(41.0, "us", "receive", 6, 1, [8.0, 8.0], [3.0, 5.0]),
        t(42.0, "us", "set", 17, feet=[3.0, 5.0]), t(43.0, "us", "attack", 10, feet=[1.0, 2.5]),
        # rally 3: our serve, their attack, our dig, #17 sets near the target, #10 kills
        t(70.0, "us", "serve"), t(71.0, "them", "receive"), t(72.0, "them", "set"),
        t(73.0, "them", "attack"), t(74.0, "us", "dig", 6, feet=[4.0, 6.0]),
        t(75.0, "us", "set", 17, feet=[5.0, 1.5]), t(76.0, "us", "attack", 10, feet=[1.2, 1.8]),
        t(100.0, "them", "serve"),                     # the next point: they serve
    ]
    spans = [(10.0, 14.0, "them", [0, 1, 2, 3], "us", "kill"),
             (40.0, 44.0, "them", [4, 5, 6, 7], "them", "attack_error"),
             (70.0, 77.0, "us", list(range(8, 15)), "us", "kill"),
             (100.0, 101.0, "them", [15], None, None)]
    rallies = [{"index": i + 1, "start_frame": int(a * 60), "end_frame": int(b * 60),
                "start_t": a, "end_t": b, "confidence": 0.9, "serving_team": s,
                "contacts": cs, "point_winner": w, "outcome": o, "outcome_confidence": 0.9}
               for i, (a, b, s, cs, w, o) in enumerate(spans)]
    write_video(conn, manifest, entry, 60.0, rallies, contacts)
    return conn


def test_setter_passer_hitter_splits(tmp_path):
    out = situational(_db(tmp_path), ["v1_set1"], SCALE, {})
    s = out["sets"]
    assert s["n"] == 3 and s["located_from"] == 3
    (setter,) = s["by_setter"]
    assert setter["jersey"] == 17 and (setter["n_in"], setter["n_out"]) == (2, 1)
    assert (setter["in"]["att"], setter["in"]["k"], setter["in"]["eff"]) == (2, 2, 1.0)
    assert (setter["out"]["e"], setter["out"]["eff"]) == (1, -1.0)
    assert {t["zone"]: t["n"] for t in setter["to"]} == {"4": 3}
    (passer,) = out["passers"]
    assert (passer["n"], passer["avg"]) == (2, 2.5)
    assert {z["zone"]: z["avg"] for z in passer["by_zone"]} == {"6": 4.0, "1": 1.0}
    (hitter,) = out["hitters"]
    assert (hitter["att"], hitter["k"], hitter["e"]) == (3, 2, 1)
    assert {b["system"]: b["att"] for b in hitter["by_system"]} == {"in": 2, "out": 1}
    assert hitter["by_setter"][0]["setter"] == 17
    (game,) = out["trends"]
    assert game["game"] == "v1" and game["points"] == "2-1"
