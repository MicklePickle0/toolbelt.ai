from setter_eyes import config
from setter_eyes.classify.rules import classify_rally
from setter_eyes.contacts.events import Contact

CFG = config.load("pipeline")["classify"]


def _t(t, team, xyz=None):
    return Contact(0, int(t * 60), t, "touch", team, -1, xyz, [0.0, 0.0], 3.0, 0.5, 1.0)


def _names(touches, serve, serving):
    return [a for a, _ in classify_rally(touches, serve, serving, CFG)]


def test_a_full_rally_by_touch_order():
    serve = _t(0.0, "us", [4.0, 9.5, 2.5])
    touches = [serve, _t(1.0, "them"), _t(2.0, "them"), _t(3.0, "them"),
               _t(3.4, "us", [4.0, 0.3, 2.8]),                    # our block at the net
               _t(4.0, "us"), _t(5.0, "us"), _t(6.0, "us"), _t(7.0, "them")]
    assert _names(touches, serve, "us") == [
        "serve", "receive", "set", "attack", "block", "dig", "set", "attack", "dig"]


def test_counted_from_the_end_when_a_touch_was_missed():
    serve = _t(0.0, "them")
    # our receive, set, attack; their dig was missed, so only set + attack are seen
    touches = [serve, _t(1.0, "us"), _t(2.0, "us"), _t(3.0, "us"),
               _t(5.0, "them"), _t(6.0, "them"), _t(7.0, "us")]
    assert _names(touches, serve, "them")[4:6] == ["set", "attack"]


def test_a_lofted_third_touch_is_a_free_ball():
    serve = _t(0.0, "them")
    touches = [serve, _t(1.0, "us"), _t(2.0, "us"), _t(3.0, "us"), _t(5.0, "them")]
    driven = classify_rally(touches, serve, "them", CFG, [None, None, None, -2.0, None])
    lofted = classify_rally(touches, serve, "them", CFG, [None, None, None, 6.0, None])
    assert driven[3][0] == "attack" and lofted[3][0] == "free_ball"


def test_coverage_after_their_block_and_overpass_on_receive():
    serve = _t(0.0, "them")
    touches = [serve, _t(1.0, "us"), _t(2.0, "us"), _t(3.0, "us"),
               _t(3.3, "them", [5.0, -0.4, 2.9]),                 # they block our attack
               _t(3.8, "us"), _t(4.8, "us"), _t(5.8, "us")]
    assert _names(touches, serve, "them")[4:] == ["block", "coverage", "set", "attack"]
    over = [serve, _t(1.0, "us"), _t(2.5, "them")]
    assert _names(over, serve, "them")[1] == "overpass"
