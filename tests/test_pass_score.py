import numpy as np

from setter_eyes import config
from setter_eyes.stats.pass_score import grade, player_position
from setter_eyes.track.people import PeopleTracks

SCALE = config.load("pass_scale")


def test_grades_follow_the_distance_from_the_target():
    tx, ty = SCALE["target"]["x"], SCALE["target"]["y"]
    on = grade(np.array([tx + 0.3, ty + 0.2]), SCALE)
    near = grade(np.array([tx + 1.8, ty + 1.2]), SCALE)
    far = grade(np.array([tx - 4.0, ty + 6.0]), SCALE)
    assert on == 4 and on > near > far and far == 1


def test_player_position_at_a_frame():
    f = np.array([100, 102, 104])
    xy = np.array([[5.0, 1.0], [5.2, 1.1], [5.4, 1.2]])
    pe = PeopleTracks(f, np.array([7, 7, 7]), np.zeros((3, 4)), np.ones(3), xy, xy,
                      np.ones(3, bool), np.zeros(3, np.int8), np.full(3, 1.8))
    np.testing.assert_allclose(player_position(pe, 7, 103), [5.2, 1.1])
    assert player_position(pe, 8, 103) is None


def test_the_setter_behind_a_back_row_player_is_found():
    from setter_eyes.ball.tracker import BallPath
    from setter_eyes.stats.pass_score import pass_spot

    f = np.array([100, 100])
    # The libero near the camera: a big box whose head is well below the ball. The setter
    # behind, partly hidden: a smaller box with the hands (box top) at the ball.
    boxes = np.array([[900.0, 700, 1150, 1300],     # libero, court (4.9, 6.0)
                      [1000.0, 560, 1080, 900]])    # setter, court (4.7, 2.3)
    xy = np.array([[4.9, 6.0], [4.7, 2.3]])
    pe = PeopleTracks(f, np.array([1, 2]), boxes, np.ones(2), xy, xy, np.ones(2, bool),
                      np.zeros(2, np.int8), np.full(2, 1.8))
    path = BallPath(np.array([100]), np.zeros(1, int), np.array([[1040.0, 565]]),
                    np.array([[1040.0, 565]]), np.ones(1, bool), np.ones(1), np.full(1, 20.0))
    np.testing.assert_allclose(pass_spot(path, pe, 100), [4.7, 2.3])            # by the image
    np.testing.assert_allclose(pass_spot(path, pe, 100, np.array([4.4, 1.0, 1.5])),
                               [4.7, 2.3])                                       # by 3D
