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
