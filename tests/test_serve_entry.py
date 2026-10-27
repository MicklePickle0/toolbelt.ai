import numpy as np

from setter_eyes import config
from setter_eyes.ball.tracker import BallPath
from setter_eyes.rally.serve_entry import find_our_serves

CFG = config.load("pipeline")["rally"]["our_serve"]
FPS, H = 60.0, 2160


def _path(events):
    """A ball path with appearances: (start s, [(dt, y_px, size_px), ...])."""
    f = np.arange(0, int(60 * FPS), 2)
    n = len(f)
    uv = np.full((n, 2), np.nan)
    size = np.full(n, np.nan)
    t = f / FPS
    for start, samples in events:
        for dt, y, s in samples:
            k = int(np.argmin(np.abs(t - (start + dt))))
            uv[k] = (1900.0, y)
            size[k] = s
    measured = np.isfinite(uv).all(axis=1)
    return BallPath(f, np.zeros(n, int), uv, uv, measured, np.ones(n), size)


def test_our_serve_is_a_big_ball_at_the_top_that_flies_away():
    served = [(0.0, 300, 150), (0.3, 500, 90), (0.8, 700, 50), (1.4, 900, 30)]
    caught = [(0.0, 300, 150), (0.3, 450, 140), (0.8, 600, 130)]       # tossed to the server
    path = _path([(10.0, served), (30.0, caught), (50.0, served)])
    touches = np.array([48.5])                       # play was on just before 50 s
    assert find_our_serves(path, FPS, touches, H, CFG) == [10.0]
