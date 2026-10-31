import numpy as np

from setter_eyes import config
from setter_eyes.ball.arcs import Arc
from setter_eyes.rally.segment import net_crossings, traced_serve

CFG = config.load("pipeline")
G = np.array([0.0, 0.0, -9.81])


def arc(t0, p0, v0, duration=0.8, conf=0.8):
    return Arc(0, 10, t0, list(p0), list(v0), 2.0, duration, 20, 1.0, conf)


def test_serve_traced_back_to_our_end_line():
    # First seen mid-court heading over the net, like a serve tracked late.
    a = arc(10.0, (5.7, 6.0, 3.2), (-1.4, -15.3, 3.0))
    t, team, p = traced_serve([a], G, 9.0, 12.0, CFG)
    assert team == "us" and t < 10.0 and 1.0 <= p[2] <= 4.2


def test_their_serve_and_non_serves():
    theirs = arc(5.0, (3.0, -5.0, 3.0), (3.6, 13.0, 2.9))
    assert traced_serve([theirs], G, 4.0, 7.0, CFG)[1] == "them"
    lob = arc(5.0, (4.0, 4.0, 1.9), (3.3, -5.4, 5.8))          # a slow free ball over the net
    assert traced_serve([lob], G, 4.0, 7.0, CFG) is None
    weak = arc(5.0, (5.7, 6.0, 3.2), (-1.4, -15.3, 3.0), conf=0.1)
    assert traced_serve([weak], G, 4.0, 7.0, CFG) is None


def test_net_crossings_count_confident_flights_only():
    over = arc(1.0, (4.5, 3.0, 3.0), (0.0, -10.0, 2.0))
    back = arc(2.0, (4.5, -5.0, 2.0), (0.0, 12.0, 3.0))
    stays = arc(3.0, (4.5, 5.0, 2.0), (0.0, 1.0, 4.0))
    weak = arc(4.0, (4.5, 3.0, 3.0), (0.0, -10.0, 2.0), conf=0.1)
    assert net_crossings([over, back, stays, weak], G, 0.0, 10.0, 0.4) == 2


def _touch(t, team, xyz=None):
    from setter_eyes.contacts.events import Contact

    return Contact(0, int(t * 60), t, "touch", team, -1, xyz, [0.0, 0.0], 3.0, 0.5, 0.8)


def _path(t_end=200.0):
    from setter_eyes.ball.tracker import BallPath

    f = np.arange(0, int(t_end * 60), 2)
    n = len(f)
    return BallPath(f, np.zeros(n, int), np.zeros((n, 2)), np.zeros((n, 2)), np.ones(n, bool),
                    np.ones(n), np.full(n, 20.0))


def test_rallies_from_touches():
    from setter_eyes.rally.segment import SERVE_INFERRED, segment_from_touches

    contacts = [
        # our server bouncing the ball well before serving: kept apart, dropped
        _touch(6.0, "us"), _touch(8.0, "us"),
        # rally 1: our serve, their reply, ours
        _touch(13.0, "us"),
        _touch(14.1, "them"), _touch(15.2, "them"), _touch(16.4, "us"),
        # rally 2: their serve unseen; our receive, set, attack, then their dig
        _touch(40.0, "us"), _touch(41.0, "us"), _touch(42.0, "us"), _touch(43.2, "them"),
        # pepper in a timeout: one side, no crossing -> no rally
        _touch(80.0, "us"), _touch(81.5, "us"), _touch(83.0, "us"),
        # a ball on the next court (x = 20 m): ignored
        _touch(120.0, "them", [20.0, 3.0, 2.0]), _touch(121.0, "us", [20.0, 5.0, 2.0]),
    ]
    rallies = segment_from_touches(_path(), [], contacts, G, 60.0, CFG)
    assert len(rallies) == 2
    r1, r2 = rallies
    assert (r1.serving_team, r1.start_t) == ("us", 13.0) and contacts[2].serve
    assert r2.serving_team == "them" and r2.serve_contact == SERVE_INFERRED
    assert r2.start_t < 40.0
    assert r2.confidence < CFG["review"]["default_min_confidence"]   # goes to review


def test_an_entry_serve_with_no_flight_is_not_a_rally():
    from setter_eyes.ball.tracker import BallPath
    from setter_eyes.rally.segment import segment_from_touches

    # Ball seen 9.8-10.2 s (lost at once) and 49.8-52 s (an ace: the serve flight followed).
    f = np.concatenate([np.arange(588, 612, 2), np.arange(2988, 3120, 2)])
    n = len(f)
    path = BallPath(f, np.zeros(n, int), np.zeros((n, 2)), np.zeros((n, 2)), np.ones(n, bool),
                    np.ones(n), np.full(n, 20.0))
    held, ace = _touch(10.0, "us"), _touch(50.0, "us")
    held.entry_serve = ace.entry_serve = True
    rallies = segment_from_touches(path, [], [held, ace], G, 60.0, CFG)
    assert [r.start_t for r in rallies] == [50.0]


def test_touches_after_the_ball_lands_are_dead():
    from setter_eyes.contacts.events import Contact
    from setter_eyes.rally.segment import segment_from_touches

    floor = Contact(0, int(15.0 * 60), 15.0, "floor", None, -1, [4.0, -5.0, 0.0], [0.0, 0.0],
                    0.0, 0.0, 0.7)
    contacts = [_touch(13.0, "us"), _touch(14.1, "them"), floor,
                _touch(15.6, "them"), _touch(16.5, "them")]   # picked up after the kill
    (r,) = segment_from_touches(_path(), [], contacts, G, 60.0, CFG)
    assert (r.end_t, r.end_reason) == (15.0, "floor")
    assert 3 not in r.contacts and 4 not in r.contacts


def _scored(t, p):
    c = _touch(t, "us" if p >= 0.5 else "them")
    c.p_ours = p
    return c


def _decode(contacts, serving):
    from setter_eyes.rally.segment import _apply_touch_prior

    _apply_touch_prior(contacts, list(range(len(contacts))), CFG, serving)
    return [c.team for c in contacts]


def test_four_in_a_row_flips_the_least_sure_touch():
    # Their serve, then our receive, set, attack and a "dig" the scorer half-believes is
    # ours: the ball went over, so the 4th is theirs.
    cs = [_scored(1.0, 0.9), _scored(1.8, 0.95), _scored(2.4, 0.9), _scored(3.0, 0.6),
          _scored(3.9, 0.2)]
    assert _decode(cs, "them") == ["us", "us", "us", "them", "them"]
    assert cs[3].confidence == 0.4 and cs[0].confidence == 0.8   # flipped: to review


def test_a_sure_fourth_touch_stays_but_never_a_fifth():
    cs = [_scored(t, 0.99) for t in (1.0, 1.8, 2.4, 3.0)]
    assert _decode(cs, "them") == ["us"] * 4                     # block + 3, allowed
    cs = [_scored(t, 0.99) for t in (1.0, 1.8, 2.4, 3.0, 3.6)]
    assert _decode(cs, "them").count("them") == 1


def test_serving_side_cannot_touch_right_after_its_serve():
    cs = [_scored(1.0, 0.7), _scored(1.6, 0.1)]
    assert _decode(cs, "us") == ["them", "them"]


def test_plausible_teams_are_left_alone():
    cs = [_scored(1.0, 0.8), _scored(1.7, 0.7), _scored(2.3, 0.9), _scored(3.0, 0.1)]
    assert _decode(cs, "them") == ["us", "us", "us", "them"]
    assert all(c.confidence == 0.8 for c in cs)
