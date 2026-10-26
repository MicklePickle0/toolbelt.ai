import numpy as np

from setter_eyes import config
from setter_eyes.ball.tracker import Detections
from setter_eyes.detect.roi import CROP, FULL, SKIP, RoiPlanner, simulate

CFG = config.load("pipeline")["detect"]["roi"]


def test_follows_the_ball_then_idles_and_backfills():
    fps = 30.0
    frames = np.arange(0, 300)                    # 10 s at 30 processed frames a second
    det_f, boxes = [], []
    for f in frames:
        t = f / fps
        if t < 3.0:                               # the ball flying right
            x = 500 + 400 * t
            det_f.append(f)
            boxes.append([x - 15, 1000, x + 15, 1030])
        elif t > 7.0:                             # it comes back after dead time
            det_f.append(f)
            boxes.append([2000, 600, 2030, 630])
    dets = Detections(frames, np.array(det_f), np.array(boxes, float),
                      np.full(len(det_f), 0.9), fps)
    out, share, modes = simulate(dets, 3840, 2160, (3, 3), 0.15, CFG)
    assert modes[CROP] > 0.4 and modes[SKIP] > 0.1 and share < 0.5
    t_out = out.det_frame / fps
    assert t_out.min() == 0.0 and ((t_out > 3.0) & (t_out < 7.0)).sum() == 0
    # The return after idle is backfilled: nothing near 7 s is lost.
    assert np.isclose(t_out[t_out > 7.0].min(), 7.0 + 1 / fps, atol=0.05)


def test_plan_falls_back_to_full_when_lost():
    p = RoiPlanner(3840, 2160, (3, 3), 0.15, CFG)
    assert p.plan(0.0)[0] == FULL                     # nothing seen yet
    p.update(0.0, np.array([[1000.0, 1000, 1030, 1030]]), np.array([0.9]))
    assert p.plan(0.033)[0] == CROP
    assert p.plan(1.0)[0] == FULL                     # lost for too long
