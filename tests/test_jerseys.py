import numpy as np

from setter_eyes.identity.numbers import NumberIndex, back_crop, stretches
from setter_eyes.identity.toucher import Poses, TouchPicker, assign_players

ROSTER = [6, 14, 17]
NUM_CFG = {"min_log_p": -1.5, "min_share": 0.6, "switch_reads": 2, "vote_prior": 0.5,
           "window_s": 4.0, "tau_s": 5.0, "conflict_factor": 0.3, "keypoint_min_score": 0.3,
           "back_min_shoulder_ratio": 0.35, "crop_side": 0.25, "crop_top": -0.05,
           "crop_bottom": 0.6}


def _read(number: int, sure: float = -0.1) -> list[float]:
    return [sure if n == number else -6.0 for n in ROSTER]


def test_a_track_handed_to_another_player_takes_the_number_read_around_the_touch():
    # pakmen-black_set2 track 0: the libero #6, then #14, then #17; one blurred "17" read
    # inside #14's time is outvoted, an implausible read doesn't count.
    frames = [0, 60, 120, 600, 660, 720, 780, 840, 1980, 2040, 2100, 900]
    numbers = [6, 6, 6, 14, 14, 17, 14, 14, 17, 17, 17, 14]
    log_p = np.array([_read(n) for n in numbers[:-1]] + [_read(14, sure=-4.0)])
    index = NumberIndex(np.zeros(12, int), np.array(frames), log_p, ROSTER, 60.0, NUM_CFG)
    assert index.number_at(0, 100)[0] == 6
    assert index.number_at(0, 720)[0] == 14
    assert index.number_at(0, 2050)[0] == 17
    assert index.number_at(0, 1400) is None               # no read within the window
    assert index.number_at(1, 700) is None                # a track with no reads
    found = stretches(np.zeros(12, int), np.array(frames), log_p, ROSTER, NUM_CFG)
    assert [s.number for s in found] == [6, 14, 17]       # the numbers.json summary


def test_one_number_is_on_one_player_at_a_time():
    # Tracks 1 and 2 on court together, both read as 14; track 1 much more often.
    frames = np.array([0, 30, 60, 90, 120, 150, 40, 100])
    tids = np.array([1] * 6 + [2] * 2)
    index = NumberIndex(tids, frames, np.array([_read(14)] * 8), ROSTER, 60.0, NUM_CFG)
    strong, weak = index.number_at(1, 80), index.number_at(2, 80)
    assert strong[0] == weak[0] == 14 and weak[1] < 0.5 * strong[1]

def test_back_crop_only_for_players_facing_away():
    kp = np.zeros((17, 2))
    score = np.ones(17)
    kp[5], kp[6], kp[11], kp[12] = (40, 100), (80, 100), (45, 180), (75, 180)
    box = back_crop(kp, score, NUM_CFG)
    assert box is not None and box[0] < 40 < 80 < box[2] and box[1] < 100 < box[3] < 180
    kp[5], kp[6] = (80, 100), (40, 100)                   # left shoulder on the right: facing us
    assert back_crop(kp, score, NUM_CFG) is None


def _pose(frame, tid, wrist, elbow, top=0.0):
    kp = np.zeros((17, 2), np.float32)
    kp[7] = kp[8] = elbow
    kp[9] = kp[10] = wrist
    return frame, tid, [0, top, 100, top + 200], kp, np.ones(17, np.float32)


def test_the_toucher_is_whose_hands_are_at_the_ball_not_the_biggest_box():
    rows = [_pose(f, 1, (50, 60), (50, 100)) for f in (0, 6)] + \
           [_pose(f, 2, (300, 300), (300, 340)) for f in (0, 6)]   # arms far from the ball
    poses = Poses(*(np.array(x) for x in zip(*rows, strict=True)))
    cfg = {"max_dt_s": 0.15, "keypoint_min_score": 0.3, "hand_extension": 0.35,
           "reach_bodies": 0.35, "base_confidence": 0.4, "fallback_factor": 0.4}
    picker = TouchPicker(poses, 60.0, cfg)
    tid, conf = picker.pick(3, (52, 45))                  # just past the wrist: the hands
    assert tid == 1 and conf > 0.5
    assert picker.pick(3, (900, 900)) is None             # nobody's arms near the ball

    class C:                                               # minimal Contact / Rally stand-ins
        def __init__(self, team, track):
            self.kind, self.team, self.inferred, self.frame = "touch", team, False, 3
            self.ball_uv, self.track_id, self.jersey, self.player_confidence = \
                (52, 45), track, None, None

    class R:
        contacts = [0, 1]

    contacts = [C("us", 2), C("them", 2)]
    index = NumberIndex(np.array([1, 1, 1]), np.array([0, 6, 12]), np.array([_read(14)] * 3),
                        ROSTER, 60.0, NUM_CFG)
    how = assign_players([R()], contacts, picker, index,
                         {"toucher": cfg, "min_player_confidence": 0.3})
    assert (contacts[0].track_id, contacts[0].jersey) == (1, 14)
    assert contacts[1].jersey is None                      # their touch: not ours to name
    assert how["number given"] == 1


def test_the_libero_is_told_apart_by_jersey_colour():
    from setter_eyes.identity.toucher import LiberoColour
    lab = np.array([[240, 128, 128], [120, 90, 160], [238, 127, 129], [122, 92, 158]],
                   np.float32)                            # white libero (1), green team (2)
    poses = Poses(np.array([0, 0, 6, 6]), np.array([1, 2, 1, 2]), np.zeros((4, 4)),
                  np.zeros((4, 17, 2)), np.ones((4, 17)), lab)
    reads = (np.array([1] * 25 + [2] * 25), np.array([0] * 25 + [6] * 25),
             np.array([6] * 25 + [14] * 25))
    cfg = {"min_reads": 20, "min_separation": 15.0, "margin": 8.0, "confidence": 0.8}
    lib = LiberoColour(poses, *reads, [6], cfg)
    assert lib.ok and lib.which(1, 3, 10) == "libero" and lib.which(2, 3, 10) == "team"
    assert not LiberoColour(poses, *reads, [], cfg).ok   # no libero on the roster
