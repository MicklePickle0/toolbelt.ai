from setter_eyes.contacts.evaluate import evaluate
from setter_eyes.labels.events import ContactLabel, RallyLabel

FPS = 60.0
CFG = {"evaluate": {"match_tolerance_s": 0.25, "rally_margin_s": 1.0}}


def _touch(t, team, conf=0.8, kind="touch"):
    return {"t": t, "team": team, "confidence": conf, "kind": kind}


def test_matches_misses_and_false_alarms():
    labels = [RallyLabel(1, 1, int(10 * FPS), int(16 * FPS), "us", contacts=[
        ContactLabel(int(10.5 * FPS), "them", "serve"),
        ContactLabel(int(12.0 * FPS), "us", "receive"),
        ContactLabel(int(13.0 * FPS), "us", "set"),
        ContactLabel(int(14.0 * FPS), "us", "attack"),
    ])]
    contacts = [
        _touch(12.1, "us"),            # receive, 0.1 s late
        _touch(13.05, "them"),         # set, wrong team
        _touch(13.6, "us"),            # nothing there: false alarm
        _touch(15.8, None, kind="floor"),
        _touch(40.0, "us"),            # outside the labelled stretch: not judged
    ]
    rallies = [{"index": 3, "start_t": 10.45, "end_t": 15.8, "serving_team": "them",
                "serve_contact": -2, "end_reason": "floor", "confidence": 0.7}]
    ev = evaluate(labels, contacts, rallies, FPS, CFG)
    s = ev.summary()
    assert s["touches_labelled"] == 4
    assert s["touches_predicted"] == 4          # 3 touches + the traced serve
    assert s["touch_recall"] == 3 / 4           # the attack was missed
    assert s["touch_precision"] == 3 / 4
    assert abs(s["team_accuracy"] - 2 / 3) < 1e-9
    assert s["recall_by_action"]["attack"] == "0/1"
    assert s["rallies_found"] == 1 and s["serves_found"] == 1
    assert s["serving_team_correct"] == 1
    assert ev.rallies[0].serve_how == "traced"


def test_each_label_matches_at_most_once():
    labels = [RallyLabel(1, 1, 0, int(5 * FPS), "us",
                         contacts=[ContactLabel(int(2.0 * FPS), "us", "dig")])]
    contacts = [_touch(1.95, "us"), _touch(2.05, "us")]
    s = evaluate(labels, contacts, [], FPS, CFG).summary()
    assert s["touch_recall"] == 1.0 and s["touch_precision"] == 0.5
    assert s["rallies_found"] == 0


def test_a_predicted_rally_in_a_timeout_is_false():
    labels = [RallyLabel(1, 1, int(10 * FPS), int(20 * FPS), "us"),
              RallyLabel(1, 2, int(90 * FPS), int(100 * FPS), "them")]
    rally = {"serve_contact": None, "serving_team": None, "end_reason": "ball_lost",
             "confidence": 0.5}
    rallies = [{**rally, "index": 1, "start_t": 11.0, "end_t": 19.0},
               {**rally, "index": 2, "start_t": 50.0, "end_t": 55.0},     # timeout pepper
               {**rally, "index": 3, "start_t": 300.0, "end_t": 310.0}]   # past the labels
    ev = evaluate(labels, [], rallies, FPS, CFG)
    assert [r["index"] for r in ev.false_rallies] == [2]
    assert ev.summary()["rallies_found"] == 1
