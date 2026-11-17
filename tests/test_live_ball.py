import numpy as np

from setter_eyes import config
from setter_eyes.rally.live import PlayerMotion, dead_from
from setter_eyes.track.people import PeopleTracks

CFG = config.load("pipeline")["rally"]["live_ball"]
FPS = 60.0


def _people(stop_at):
    """Six players in our half running about until `stop_at` s, then standing still."""
    frames = np.arange(0, int(20 * FPS), 2)
    rows = []
    for k in range(6):
        for f in frames:
            t = f / FPS
            moving = min(t, stop_at)
            rows.append((f, k, 2 + k, 3 + np.sin(moving * 2.0 + k) * 2))
    a = np.array(rows)
    n = len(a)
    xy = a[:, 2:4]
    return PeopleTracks(a[:, 0].astype(int), a[:, 1].astype(int), np.zeros((n, 4)), np.ones(n),
                        xy, xy, np.ones(n, bool), np.zeros(n, np.int8), np.full(n, 1.8))


def test_a_touch_after_play_stopped_is_dead():
    motion = PlayerMotion(_people(stop_at=10.0), FPS)
    touches = [5.0, 6.1, 7.2, 8.3, 11.5]          # the last: 3.2 s later, players standing
    assert dead_from(touches, 5.0, motion, CFG) == 4


def test_a_late_touch_in_live_play_is_kept():
    motion = PlayerMotion(_people(stop_at=99.0), FPS)
    assert dead_from([5.0, 6.1, 7.2, 8.3, 11.5], 5.0, motion, CFG) == 5
    assert dead_from([5.0, 6.1], 5.0, None, CFG) == 2
