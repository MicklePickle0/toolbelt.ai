from setter_eyes import config
from setter_eyes.labels.events import ContactLabel, RallyLabel
from setter_eyes.stats.passing import passing_average, receptions

SCALE = config.load("pass_scale")


def _rally(n, contacts, winner, outcome=None):
    return RallyLabel(1, n, 0, 100, winner, outcome,
                      [ContactLabel(10 * i, team, action, jersey, score)
                       for i, (team, action, jersey, score) in enumerate(contacts)])


def test_an_overpass_on_serve_receive_is_a_zero_pass():
    rallies = [
        _rally(1, [("them", "serve", None, None), ("us", "receive", 7, 3),
                   ("us", "set", 1, None), ("us", "attack", 13, None)], "us", "kill"),
        _rally(2, [("them", "serve", None, None), ("us", "overpass", 7, None),
                   ("them", "attack", None, None)], "them", "kill"),
        _rally(3, [("them", "serve", None, None)], "them", "ace"),          # untouched ace
        _rally(4, [("us", "serve", 5, None), ("them", "receive", None, None),
                   ("us", "dig", 7, None)], "us"),                          # our serve: none
        _rally(5, [("them", "serve", None, None), ("us", "receive", 9, 4)], "us"),
    ]
    recs = receptions(rallies, SCALE)
    assert [(r.rally, r.jersey, r.score, r.how) for r in recs] == [
        (1, 7, 3, "receive"), (2, 7, 0, "overpass"), (3, None, 0, "ace"), (5, 9, 4, "receive")]
    avg = passing_average(recs)
    assert avg[7] == (2, 1.5)                     # the overpass pulls #7 from 3.0 to 1.5
    assert avg[9] == (1, 4.0)
    assert avg["team"] == (4, 1.75)               # the ace counts for the team only


def test_a_dig_after_their_attack_is_not_a_reception():
    r = _rally(1, [("them", "serve", None, None), ("us", "receive", 7, 2),
                   ("us", "set", 1, None), ("us", "free_ball", 4, None),
                   ("them", "attack", None, None), ("us", "dig", 7, None)], "us")
    assert [(x.jersey, x.score) for x in receptions([r], SCALE)] == [(7, 2)]


def test_a_replayed_rally_counts_for_nothing():
    from setter_eyes.stats.actions import counts

    replay = _rally(1, [("them", "serve", None, None), ("us", "receive", 7, 1)], None, "replay")
    assert receptions([replay], SCALE) == []
    assert counts([replay]) == {}
