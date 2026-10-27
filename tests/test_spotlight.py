import numpy as np

from setter_eyes import config
from setter_eyes.ball.tracker import BallPath
from setter_eyes.track.people import PeopleTracks
from setter_eyes.track.spotlight import draw_ring, follow, toucher_track
from test_calibration import look_at

CFG = config.load("pipeline")["overlay"]["spotlight"]


def _people(rows):
    """rows: (frame, track id, court x, court y)."""
    a = np.array(rows, float)
    n = len(a)
    xy = a[:, 2:4]
    return PeopleTracks(a[:, 0].astype(int), a[:, 1].astype(int),
                        np.tile([100.0, 100, 150, 300], (n, 1)), np.ones(n), xy, xy,
                        np.ones(n, bool), np.zeros(n, np.int8), np.full(n, 1.8))


def test_our_attacker_not_their_blocker():
    people = _people([(100, 1, 0.9, 1.5),       # our attacker, back from the net
                      (100, 2, 1.6, -0.3)])     # their blocker, just across it
    path = BallPath(np.array([100]), np.zeros(1, int), np.zeros((1, 2)), np.zeros((1, 2)),
                    np.ones(1, bool), np.ones(1), np.full(1, 20.0))
    assert toucher_track(path, people, 100, xyz=np.array([2.1, 0.6, 2.4])) == 1


def test_follow_continues_a_broken_track():
    rows = [(f, 1, 3.0, 4.0 + 0.01 * f) for f in range(0, 60, 2)]
    rows += [(f, 2, 3.1, 4.6 + 0.01 * (f - 70)) for f in range(70, 120, 2)]   # same player
    rows += [(f, 3, 8.0, 1.0) for f in range(70, 120, 2)]                     # someone else
    people = _people(rows)
    chain = follow(people, 1, 0, 118, fps=60.0)
    assert set(people.track_id[chain]) == {1, 2}


def test_ring_is_a_band_around_the_feet():
    cam = look_at((4.5, 13.0, 1.6), (4.5, 0.0, 0.6))
    img = np.zeros((cam.height, cam.width, 3), np.uint8)
    xy = np.array([4.5, 4.0])
    draw_ring(img, cam, xy, None, CFG)
    uv, _ = cam.project(np.array([[4.5, 4.0, 0.0]]))
    u, v = uv[0].round().astype(int)
    assert img[v, u].sum() == 0                          # the middle (the feet) is clear
    assert img.sum() > 0                                 # the band is drawn
