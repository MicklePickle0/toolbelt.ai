from setter_eyes.labels.events import ContactLabel, RallyLabel
from setter_eyes.stats.actions import counts, stat_actions


def _rally(contacts, winner, outcome=None):
    return RallyLabel(1, 1, 0, 100, winner, outcome,
                      [ContactLabel(10 * i, team, action, jersey)
                       for i, (team, action, jersey) in enumerate(contacts)])


def test_a_dig_called_for_handling_is_not_a_dig():
    r = _rally([("us", "serve", 5), ("them", "receive", None), ("them", "set", None),
                ("them", "attack", None), ("us", "dig", 7)], "them", "ball_handling_error")
    assert stat_actions(r)[-1][1] == "ball_handling_error"


def test_a_clean_dig_before_a_called_set_still_counts():
    r = _rally([("us", "serve", 5), ("them", "attack", None), ("us", "dig", 7),
                ("us", "set", 1)], "them", "ball_handling_error")
    assert [a for _, a in stat_actions(r)][-2:] == ["dig", "set"]


def test_a_dig_after_their_free_ball_is_a_free_ball_pass():
    r = _rally([("us", "serve", 5), ("them", "receive", None), ("them", "free_ball", None),
                ("us", "dig", 7), ("us", "set", 1), ("us", "attack", 13)], "us", "kill")
    assert counts([r])[7] == {"free_ball_pass": 1}
