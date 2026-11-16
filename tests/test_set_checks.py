"""Match format (match.py, roadmap E1) and score/stat sanity checks (rally/sanity.py, E2)."""

from __future__ import annotations

import pytest

from setter_eyes import config, match
from setter_eyes.rally.sanity import check_set

CFG = config.load("pipeline")


# ---- match format --------------------------------------------------------------------------
def test_presets_best_of_3_and_5():
    p = match.presets(CFG)
    assert p["best_of_3"].points == (25, 25, 15)
    assert p["best_of_5"].points == (25, 25, 25, 25, 15)
    assert match.default(CFG).preset == CFG["match"]["default"]


def test_set_over_targets_and_win_by():
    f = match.parse("best_of_5", CFG)
    assert not f.set_over(24, 23, 1)
    assert f.set_over(25, 23, 1)
    assert not f.set_over(25, 24, 1)          # win by 2
    assert f.set_over(30, 28, 4)
    assert f.set_over(15, 13, 5)              # fifth set to 15
    assert not f.set_over(15, 13, 4)


def test_cap_over_target():
    f = match.parse("best_of_3_capped", CFG)
    assert f.set_over(27, 26, 1)              # 25 + 2 wins outright
    assert not f.set_over(26, 25, 1)
    assert f.set_over(17, 16, 3)              # 15 + 2 in the third set


def test_custom_format_and_errors():
    f = match.parse({"points": [21, 21, 15], "win_by": 2}, CFG)
    assert f.preset is None and f.target(2) == 21 and f.best_of == 3
    with pytest.raises(match.MatchFormatError):
        match.parse({"points": [25, 25]}, CFG)          # an even number of sets
    with pytest.raises(match.MatchFormatError):
        match.parse("best_of_7", CFG)


def test_per_game_file_and_set_numbers(tmp_path):
    path = tmp_path / "match_formats.yaml"
    match.save(path, {"games": {"playoff": "best_of_5"}, "set_numbers": {"odd_key": 4}})
    data = match.load(path)
    assert match.format_for("playoff", data, CFG).best_of == 5
    assert match.format_for("other", data, CFG).preset == CFG["match"]["default"]
    assert match.set_number("pakmen-black_set3", data) == 3
    assert match.set_number("odd_key", data) == 4
    assert match.set_number("practice", data) == 1


# ---- sanity checks -------------------------------------------------------------------------
def _rally(i, winner, serving, touches=None, conf=0.9, reviewed=0, outcome=None, gap=15.0):
    t0 = i * (8.0 + gap)
    other = "them" if serving == "us" else "us"
    if touches is None:                    # the receiving side attacks: a kill or an error
        touches = [(serving, "serve"), (other, "receive"), (other, "set"), (other, "attack")]
    if outcome is None:
        outcome = "kill" if winner == other else "attack_error"
    return {"id": i + 1, "rally_number": i + 1, "start_t": t0, "end_t": t0 + 8.0,
            "segment_confidence": conf, "serving_team": serving, "point_winner": winner,
            "point_winner_source": "user", "outcome": outcome, "reviewed": reviewed,
            "touches": [{"team": t, "action": a} for t, a in touches]}


def _set(winners: list[str | None], first_server="us", **kw):
    out, server = [], first_server
    for i, w in enumerate(winners):
        out.append(_rally(i, w, server, **kw))
        server = w or server
    return out


def test_legal_set_end_is_ok():
    rallies = _set(["us"] * 25)
    res = check_set(rallies, match.parse("best_of_3", CFG), 1, CFG)
    assert res["status"] == "ok" and res["score"] == [25, 0]
    assert res["flagged"] == []


def test_past_end_flags_extra_and_least_sure():
    winners = ["us"] * 25 + ["them"] * 3
    rallies = _set(winners)
    rallies[5]["segment_confidence"] = 0.1          # the least sure: maybe a false rally
    res = check_set(rallies, match.parse("best_of_3", CFG), 1, CFG)
    assert res["status"] == "past_end"
    codes = {rid: {f["code"] for f in c["flags"]} for rid, c in res["rallies"].items()}
    assert all("past_end" in codes[r["id"]] for r in rallies[25:])
    assert "maybe_false" in codes[rallies[5]["id"]]
    assert res["flagged"][0] in {rallies[5]["id"], *(r["id"] for r in rallies[25:])}


def test_short_set_and_last_point_unknown():
    res = check_set(_set(["us"] * 20), match.parse("best_of_3", CFG), 1, CFG)
    assert res["status"] == "short"
    res = check_set(_set(["us"] * 24 + [None]), match.parse("best_of_3", CFG), 1, CFG)
    assert res["status"] == "last_point_unknown"


def test_third_set_is_to_15():
    res = check_set(_set(["us"] * 15), match.parse("best_of_3", CFG), 3, CFG)
    assert res["status"] == "ok"
    res = check_set(_set(["us"] * 15), match.parse("best_of_5", CFG), 3, CFG)
    assert res["status"] == "short"


def test_serve_order_break_and_missed_point():
    rallies = _set(["us", "us", "them"])
    rallies[1]["serving_team"] = "them"              # stored team and serve touch agree...
    rallies[1]["touches"][0]["team"] = "them"        # ...but we won the point before
    rallies[1]["touches"][1]["team"] = "us"
    res = check_set(rallies, match.parse("best_of_3", CFG), 1, CFG)
    assert any(f["code"] == "serve_order" for f in res["rallies"][2]["flags"])
    rallies[1]["start_t"] += 60.0                    # after a long gap: a missed point
    rallies[1]["end_t"] += 60.0
    rallies[2]["start_t"] += 60.0
    rallies[2]["end_t"] += 60.0
    res = check_set(rallies, match.parse("best_of_3", CFG), 1, CFG)
    assert [f["code"] for f in res["rallies"][2]["flags"]] == ["missed_point"]


def test_serving_team_wrong_when_serve_touch_disagrees():
    rallies = _set(["them", "them"], first_server="them")
    rallies[1]["serving_team"] = "us"                # the pipeline's guess, left in review
    res = check_set(rallies, match.parse("best_of_3", CFG), 1, CFG)
    flags = res["rallies"][2]["flags"]
    assert flags[0]["code"] == "serving_team_wrong"
    assert "the point before went to them" in flags[0]["text"]


def test_touch_rules():
    r = _rally(0, "us", "them", touches=[("them", "serve"), ("us", "receive"), ("us", "set"),
                                         ("us", "dig"), ("us", "attack")])
    r2 = _rally(1, "us", "us", touches=[("us", "serve"), ("us", "receive")])
    r3 = _rally(2, "us", "us", touches=[("us", "serve"), ("them", "receive"), ("them", "set"),
                                        ("them", "attack"), ("us", "block"), ("us", "dig"),
                                        ("us", "set"), ("us", "attack")])
    res = check_set([r, r2, r3], match.parse("best_of_3", CFG), 1, CFG)
    codes = {rid: {f["code"] for f in c["flags"]} for rid, c in res["rallies"].items()}
    assert "four_touches" in codes[1]
    assert "receive_on_own_serve" in codes[2]
    assert "four_touches" not in codes[3]            # a block doesn't count


def test_outcome_side():
    r = _rally(0, "them", "us", outcome="ace", touches=[("us", "serve")])
    res = check_set([r], match.parse("best_of_3", CFG), 1, CFG)
    assert res["rallies"][1]["flags"][-1]["code"] == "outcome_side"


def test_reviewed_rallies_rank_by_flags_not_confidence():
    rallies = _set(["us"] * 3, reviewed=1, conf=0.1)
    res = check_set(rallies, match.parse("best_of_3", CFG), 1, CFG)
    assert all(c["priority"] == 1.0 for c in res["rallies"].values())
