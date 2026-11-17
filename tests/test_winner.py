from setter_eyes import config
from setter_eyes.contacts.events import Contact
from setter_eyes.rally.segment import Rally
from setter_eyes.rally.winner import infer_winners

CFG = config.load("pipeline")


def _rally(i, t0, serving, contacts, conf=0.8):
    return Rally(i, int(t0 * 60), int((t0 + 6) * 60), t0, t0 + 6.0, serving, None, "floor",
                 contacts, 1, conf)


def _touch(t, team, action):
    c = Contact(0, int(t * 60), t, "touch", team, -1, None, [0.0, 0.0], 3.0, 0.5, 0.9)
    c.action, c.action_confidence = action, 0.9
    return c


def test_the_next_server_won_the_point():
    contacts = [_touch(10, "us", "serve"), _touch(12, "them", "receive"),
                _touch(14, "them", "attack"),                      # their kill...
                _touch(30, "them", "serve"),                       # ...so they serve next
                _touch(32, "us", "receive"), _touch(34, "us", "attack"),
                _touch(80, "us", "serve")]                         # 40 s gap: a doubt
    rallies = [_rally(1, 10, "us", [0, 1, 2]), _rally(2, 30, "them", [3, 4, 5]),
               _rally(3, 80, "us", [6])]
    infer_winners(rallies, contacts, CFG)
    r1, r2, r3 = rallies
    assert (r1.point_winner, r1.outcome, r1.score_us, r1.score_them) == ("them", "kill", 0, 1)
    assert (r2.point_winner, r2.outcome) == ("us", "kill")
    assert r2.winner_confidence < r1.winner_confidence             # long gap to rally 3
    assert r3.point_winner is None and r3.outcome is None          # no next serve
