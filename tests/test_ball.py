import numpy as np

from setter_eyes import config
from setter_eyes.ball.arcs import fit_flights, segment
from setter_eyes.ball.kalman import BallKalman
from setter_eyes.ball.tracker import Detections, stabilise, track_ball, unstabilise
from setter_eyes.calibration.camera import Camera
from test_calibration import VIEWS, look_at

CFG = config.load("pipeline")["ball"]
FPS = 59.94
G = np.array([0.0, 0.0, -9.81])


def flights():
    """A serve over the net, then a touch sending it back: (frames, 3D truth)."""
    a0, va = np.array([4.5, 10.0, 2.6]), np.array([0.3, -16.0, 3.5])
    ta = np.arange(0, 1.0, 2 / FPS)
    pa = a0 + va * ta[:, None] + 0.5 * G * ta[:, None] ** 2
    b0 = a0 + va * 1.0 + 0.5 * G
    vb = np.array([-1.0, 12.0, 6.0])
    tb = np.arange(2 / FPS, 1.0, 2 / FPS)
    pb = b0 + vb * tb[:, None] + 0.5 * G * tb[:, None] ** 2
    pts = np.vstack([pa, pb])
    frames = 1000 + 2 * np.arange(len(pts))
    return frames, pts, len(pa)


def synthetic_detections(cam: Camera, seed=0):
    rng = np.random.default_rng(seed)
    frames, pts, n_a = flights()
    uv, _ = cam.project(pts)
    size = cam.f * 0.21 / np.linalg.norm(pts - cam.centre, axis=1)
    det_f, det_b, det_s = [], [], []
    for f, (u, v), s in zip(frames, uv, size, strict=True):
        if rng.random() > 0.15:                                   # 15% missed detections
            u, v = u + rng.normal(0, 1.0), v + rng.normal(0, 1.0)
            det_f.append(f)
            det_b.append([u - s / 2, v - s / 2, u + s / 2, v + s / 2])
            det_s.append(0.9)
        det_f.append(f)                                           # a spare ball that never moves
        det_b.append([195, 595, 205, 605])
        det_s.append(0.8)
        if rng.random() < 0.2:                                    # low-score clutter
            x, y = rng.uniform(0, cam.width), rng.uniform(0, cam.height)
            det_f.append(f)
            det_b.append([x - 5, y - 5, x + 5, y + 5])
            det_s.append(rng.uniform(0.2, 0.45))
    dets = Detections(frames.copy(), np.array(det_f), np.array(det_b, float), np.array(det_s),
                      FPS)
    return dets, frames, pts, uv, n_a


def test_stabilise_round_trip_and_cancels_pan():
    ref = VIEWS["centred"]
    panned = look_at((4.5, 13.0, 1.6), (5.5, 0.0, 0.8))         # same centre, turned
    p = np.array([[4.0, 2.0, 3.0]])
    uv_ref = ref.project(p)[0]
    uv_pan = panned.project(p)[0]
    assert np.linalg.norm(uv_ref - uv_pan) > 30                   # the pan moves the pixel
    assert np.allclose(stabilise([panned], ref, uv_pan), uv_ref, atol=0.05)
    assert np.allclose(unstabilise(panned, ref, uv_ref), uv_pan, atol=0.05)


def test_kalman_bridges_a_gap_on_a_parabola():
    t = np.arange(0, 0.6, 1 / 30)
    uv = np.stack([400 + 900 * t, 300 - 1200 * t + 0.5 * 1800 * t**2], axis=1)
    kf = BallKalman(uv[0], t[0], CFG["kalman"], 1800.0)
    for ti, p in zip(t[1:12], uv[1:12], strict=True):
        kf.predict(ti)
        kf.update(p, 2.0)
    pos, _ = kf.peek(t[15])                                       # three samples ahead
    assert np.linalg.norm(pos - uv[15]) < 8


def test_tracker_follows_the_ball_not_the_spare():
    cam = VIEWS["centred"]
    dets, frames, pts, uv, _ = synthetic_detections(cam)
    path = track_ball(dets, lambda f: cam, cam, CFG)
    ok = np.isfinite(path.uv).all(axis=1)
    assert ok.mean() > 0.9                                        # gaps were bridged
    err = np.linalg.norm(path.uv[ok] - uv[ok], axis=1)
    assert np.median(err) < 3 and (err < 15).mean() > 0.95
    spare = np.linalg.norm(path.uv[ok] - [200, 600], axis=1) < 20
    assert not spare.any()


def test_flights_are_split_at_the_touch_and_fit_in_3d():
    cam = VIEWS["centred"]
    dets, frames, pts, uv, n_a = synthetic_detections(cam, seed=1)
    path = track_ball(dets, lambda f: cam, cam, CFG)
    times = path.frames / FPS
    valid = path.measured & np.isfinite(path.uv).all(axis=1)
    arcs = fit_flights(times, path.uv_ref, path.uv, valid, [cam] * len(path.frames),
                       path.size_px, CFG)
    long = [a for a in arcs if a.duration_s > 0.5]
    assert len(long) == 2
    touch = frames[n_a] / FPS
    assert abs(long[0].t0 + long[0].duration_s - touch) < 0.15     # split near the touch
    for arc in long:
        sel = np.arange(arc.first, arc.last + 1)
        est = arc.position(times[sel], G)
        assert np.linalg.norm(est - pts[sel], axis=1).max() < 0.6  # metres, at 1280 px
        assert arc.confidence > CFG["arcs"]["min_confidence"]
        assert 0.7 < arc.size_ratio < 1.3


def test_segment_rejects_noise_shorter_than_a_flight():
    t = np.arange(0, 0.15, 1 / 30)
    assert segment(t, np.zeros((len(t), 2)), CFG["arcs"]) == []
