import numpy as np

from conftest import FPS as VIDEO_FPS
from conftest import N_FRAMES, decoded_index
from setter_eyes import config
from setter_eyes.datasets.tiles import tile_boxes
from setter_eyes.detect.follow import BallFollower
from setter_eyes.detect.video import detect_video

W, H = 3840, 2160
FPS = 30.0
GRID, OVERLAP = (3, 3), 0.15
CFG = config.for_video(config.load("pipeline"), W)["detect"]["ball_follow"]


def fake_windows(scene):
    """A detector that finds each ball of scene(frame) whose centre is inside a window."""
    state = {"frame": 0}

    def detect(_bgr, windows):
        out = []
        for (u, v), score in scene(state["frame"]):
            if any(x <= u < x + w and y <= v < y + h for x, y, w, h in windows):
                out.append(([u - 10, v - 10, u + 10, v + 10], score))
        return out

    return detect, state


def run(scene, n=60):
    detect, state = fake_windows(scene)
    follower = BallFollower(detect, GRID, OVERLAP, FPS, CFG)
    image = np.zeros((H, W, 3), np.uint8)
    found = []
    for f in range(n):
        state["frame"] = f
        found.append(follower(image, f))
    return follower, found


def flying(f):
    return np.array([400.0 + 30 * f, 1500.0 - 20 * f + 0.4 * f * f])


def test_following_finds_the_ball_every_frame_with_few_passes():
    spare = (np.array([3000.0, 1900.0]), 0.8)                     # never moves

    def scene(f):
        return [(flying(f), 0.9), spare]

    follower, found = run(scene)
    for f, dets in enumerate(found):
        centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, _ in dets]
        assert any(np.hypot(u - flying(f)[0], v - flying(f)[1]) < 1 for u, v in centres), f
    n_tiles = len(tile_boxes(W, H, GRID, OVERLAP))
    assert follower.passes / follower.frames < n_tiles / 3        # a 3x+ saving
    assert follower.full_scans < follower.frames / 4


def test_a_window_that_misses_its_ball_scans_the_same_frame_in_full():
    def scene(f):
        p = flying(f) if f < 40 else flying(f) + [-2500.0, 0.0]   # a sharp jump at f = 40
        return [(np.clip(p, 0, [W - 1, H - 1]), 0.9)]

    _, found = run(scene, n=60)
    assert all(len(d) == 1 for d in found)                       # never lost a sample


def test_only_spare_balls_means_full_scans():
    def scene(_f):
        return [(np.array([3000.0, 1900.0]), 0.8), (np.array([200.0, 300.0]), 0.7)]

    follower, found = run(scene, n=30)
    assert follower.full_scans == follower.frames
    assert all(len(d) == 2 for d in found)


def test_new_ball_is_confirmed_with_full_scans_then_followed():
    def scene(f):
        return [(flying(f), 0.9)] if f >= 10 else []

    follower, found = run(scene, n=40)
    assert all(len(found[f]) == 1 for f in range(10, 40))


def test_settings_scale_with_resolution():
    full = config.load("pipeline")["detect"]["ball_follow"]
    half = config.for_video(config.load("pipeline"), 1920)["detect"]["ball_follow"]
    assert half["gate_px"] == full["gate_px"] / 2
    assert half["min_score"] == full["min_score"]                 # not a pixel setting


def test_detect_video_skips_frames_outside_keep_and_resumes(video, tmp_path):
    out = tmp_path / "dets.npz"
    seen = []

    def detect(image, index):
        assert decoded_index(image) == index
        seen.append(index)
        return [([0, 0, 4, 4], 0.9)]

    def keep(f):
        return 10 <= f < 20 or f >= 40

    common = dict(stride=2, threshold=0.1, fps=VIDEO_FPS, total_frames=N_FRAMES,
                  keep=keep, log=lambda _m: None)
    detect_video(video, out, detect, last=30, settings="a", **common)
    assert seen == list(range(10, 20, 2))
    detect_video(video, out, detect, settings="a", **common)       # resumes after frame 18
    assert seen == list(range(10, 20, 2)) + list(range(40, N_FRAMES, 2))
    with np.load(out) as d:
        assert list(d["frames"]) == seen and str(d["settings"]) == "a"

    seen.clear()
    detect_video(video, out, detect, last=30, settings="b", **common)   # other settings
    assert seen == list(range(10, 20, 2))                           # started over


def test_a_run_from_before_settings_were_stored_is_continued(video, tmp_path):
    out = tmp_path / "old.npz"
    np.savez_compressed(out, frames=np.array([0, 2, 4]), det_frame=np.array([], np.int64),
                        det_box=np.zeros((0, 4), np.float32),
                        det_score=np.zeros(0, np.float32), stride=2, threshold=0.1, fps=30.0)
    seen = []
    detect_video(video, out, lambda _i, f: seen.append(f) or [], stride=2, threshold=0.1,
                 fps=VIDEO_FPS, total_frames=N_FRAMES, last=12, settings="follow=...",
                 log=lambda _m: None)
    assert seen == [6, 8, 10]
