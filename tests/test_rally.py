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
