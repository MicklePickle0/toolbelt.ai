import cv2
import numpy as np
import pytest

from setter_eyes import config
from setter_eyes.calibration import (
    Calibration,
    Camera,
    auto_calibrate,
    court_model,
    draw_court,
    line_mask,
    load,
    median_background,
    refine_on_lines,
    save,
    solve_from_points,
)
from setter_eyes.calibration.camera import to_cv
from setter_eyes.calibration.fit import focal_from_hfov
from setter_eyes.calibration.lines import LineScorer

W, H = 1280, 720
CFG = config.load("court")["calibration"]
MODEL = court_model(2.43)


def look_at(centre, target, hfov=70.0, k1=0.0) -> Camera:
    """Camera at `centre` looking at `target` (court coordinates), no roll."""
    c, t = to_cv(np.array([centre]))[0], to_cv(np.array([target]))[0]
    fwd = (t - c) / np.linalg.norm(t - c)
    x = np.cross([0, 0, -1.0], fwd)
    x /= np.linalg.norm(x)
    y = np.cross(fwd, x)
    R = np.array([x, y, fwd])
    return Camera(W, H, focal_from_hfov(W, hfov), cv2.Rodrigues(R)[0], -R @ c, k1)


def render(cam: Camera, seed=0, distractors=True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (120, 165, 205), np.uint8)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
    if distractors:
        for _ in range(6):   # basketball-style dark lines and a coloured arc
            p, q = rng.integers(0, [W, H], 2), rng.integers(0, [W, H], 2)
            cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), (30, 30, 30), 3)
        cv2.ellipse(img, (W // 3, int(H * 0.8)), (220, 60), 0, 180, 360, (0, 200, 230), 4)
        cv2.circle(img, (int(W * 0.7), int(H * 0.75)), 40, (255, 255, 255), -1)   # reflection
    for a, b in MODEL.lines.values():
        uv, z = cam.project(np.linspace(a, b, 200))
        ok = z > 0.1
        cv2.polylines(img, [np.round(uv[ok]).astype(np.int32)], False, (245, 245, 245), 3,
                      cv2.LINE_AA)
    return img


def keypoint_error(cam: Camera, truth: Camera) -> float:
    pts = np.array(list(MODEL.keypoints.values()))
    a, za = cam.project(pts)
    b, zb = truth.project(pts)
    ok = (zb > 0) & (b[:, 0] >= 0) & (b[:, 0] < W) & (b[:, 1] >= 0) & (b[:, 1] < H)
    return float(np.linalg.norm(a[ok] - b[ok], axis=1).max())


VIEWS = {
    "centred": look_at((4.5, 13.0, 1.6), (4.5, 0.0, 0.6)),
    "diagonal": look_at((1.0, 12.5, 1.2), (5.5, 0.5, 0.8), hfov=65),
    "wide_low": look_at((6.0, 11.0, 0.9), (4.0, 0.0, 1.0), hfov=85),
}


def test_court_frame_handedness():
    cam = VIEWS["centred"]
    uv, z = cam.project(np.array([MODEL.keypoints["near_left_corner"],
                                  MODEL.keypoints["near_right_corner"],
                                  MODEL.keypoints["net_tape_left"]]))
    assert (z > 0).all()
    assert uv[0, 0] < uv[1, 0]                     # court x increases to the right
    assert uv[2, 1] < uv[0, 1]                     # the net tape is above the floor
    assert np.allclose(cam.centre, [4.5, 13.0, 1.6])
    assert np.allclose(Camera.from_dict(cam.to_dict()).project(MODEL.lines["centre"])[0],
                       cam.project(MODEL.lines["centre"])[0])


def test_solve_from_points_recovers_focal_and_pose():
    truth = VIEWS["diagonal"]
    names = ["near_left_attack", "near_right_attack", "centre_left", "centre_right",
             "far_left_attack", "far_right_attack", "net_tape_left", "net_tape_right"]
    world = np.array([MODEL.keypoints[n] for n in names])
    image = truth.project(world)[0] + np.random.default_rng(1).normal(0, 0.5, (len(names), 2))
    cam, rms = solve_from_points(world, image, W, H)
    assert rms < 1.0
    assert cam.f == pytest.approx(truth.f, rel=0.03)
    assert np.linalg.norm(cam.centre - truth.centre) < 0.15


@pytest.mark.parametrize("view", list(VIEWS))
def test_auto_calibrate_synthetic(view):
    truth = VIEWS[view]
    image = render(truth)
    result = auto_calibrate(image, MODEL, CFG)
    assert result is not None
    assert keypoint_error(result.camera, truth) < 4.0
    assert result.support.confidence >= CFG["auto_min_confidence"]


def test_refine_recovers_from_perturbation_and_distortion():
    truth = look_at((4.5, 13.0, 1.6), (4.5, 0.0, 0.6), k1=-0.08)
    mask, s = line_mask(render(truth, distractors=False), CFG)
    scorer = LineScorer(mask, s, MODEL, CFG)
    start = Camera(W, H, truth.f * 1.05, truth.rvec + 0.01, truth.tvec + [0.2, -0.1, 0.3])
    assert keypoint_error(start, truth) > 15
    cam = refine_on_lines(start, scorer, CFG, estimate_k1=True)
    assert keypoint_error(cam, truth) < 3.0
    assert scorer.evaluate(cam).confidence > scorer.evaluate(start).confidence


def test_unrelated_image_has_low_confidence():
    rng = np.random.default_rng(3)
    img = np.full((H, W, 3), 128, np.uint8)
    for _ in range(25):
        p, q = rng.integers(0, [W, H], 2), rng.integers(0, [W, H], 2)
        cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), (250, 250, 250), 3)
    result = auto_calibrate(img, MODEL, CFG)
    assert result is None or result.support.confidence < CFG["auto_min_confidence"]


def test_median_background_removes_movers():
    base = np.full((40, 40, 3), 100, np.uint8)
    frames = []
    for i in range(5):
        f = base.copy()
        f[i * 8:i * 8 + 8, 10:20] = 255
        frames.append(f)
    assert (median_background(frames) == 100).all()


def test_store_round_trip_and_overlay(tmp_path):
    cam = VIEWS["centred"]
    cal = Calibration(video="v", camera=cam, method="manual", confidence=0.9,
                      net_height_m=2.43, clicks={"centre_left": [1.0, 2.0]})
    save(cal, tmp_path)
    back = load("v", tmp_path)
    assert back.clicks == cal.clicks and back.camera.f == pytest.approx(cam.f)
    assert load("missing", tmp_path) is None
    out = draw_court(np.zeros((H, W, 3), np.uint8), cam, MODEL, label="test")
    assert out.any()


def _pan(deg_yaw: float, deg_tilt: float) -> np.ndarray:
    return cv2.Rodrigues(np.radians([deg_tilt, deg_yaw, 0.0]))[0]


def test_rotation_between_ignores_movers():
    from setter_eyes.calibration.track import _angle_deg, rotation_between

    rng = np.random.default_rng(0)
    R = _pan(3.0, -1.0)
    a = rng.uniform(-0.6, 0.6, (300, 2))
    b_h = np.hstack([a, np.ones((300, 1))]) @ R.T
    b = b_h[:, :2] / b_h[:, 2:]
    b[:60] += rng.uniform(-0.1, 0.1, (60, 2))       # 20% "players"
    est, inliers = rotation_between(a, b, 800.0, 1.5)
    assert _angle_deg(est, R) < 0.05 and 0.75 < inliers < 0.85


def test_track_rotation_follows_a_pan():
    from setter_eyes.calibration.track import _angle_deg, track_rotation
    from setter_eyes.ingest import Frame

    base = look_at((4.5, 13.0, 1.6), (4.5, 0.0, 0.6))
    rng = np.random.default_rng(5)
    tw, th = 2 * W, 2 * H
    tex = np.full((th, tw, 3), 90, np.uint8)
    for _ in range(900):
        x, y = rng.integers(0, tw), rng.integers(0, th)
        cv2.rectangle(tex, (int(x), int(y)), (int(x + rng.integers(5, 40)),
                      int(y + rng.integers(5, 40))), rng.integers(0, 255, 3).tolist(), -1)
    K = base.K
    K_tex = K.copy()
    K_tex[0, 2], K_tex[1, 2] = tw / 2, th / 2
    n, ref = 150, 40
    pans = [_pan(12 * np.sin(i / 25), 2 * np.sin(i / 40)) for i in range(n)]

    def frames():
        for i, P in enumerate(pans):
            M = K_tex @ P.T @ np.linalg.inv(K)          # frame pixel -> texture pixel
            img = cv2.warpPerspective(tex, M, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            yield Frame(i, i / 60, img)

    # The calibrated pose is the base pose seen at the reference frame.
    cam_ref = Camera(W, H, base.f, cv2.Rodrigues(pans[ref] @ base.R)[0],
                     pans[ref] @ base.tvec)
    cal = Calibration(video="syn", camera=cam_ref, method="manual", confidence=1.0,
                      net_height_m=2.43, frame=ref)
    cfg = dict(CFG, track=dict(CFG["track"], width=W))
    track = track_rotation(frames(), cal, cfg)
    errs = [_angle_deg(track.camera(i).R, pans[i] @ base.R) for i in range(n)]
    assert max(errs) < 0.3, max(errs)
    assert np.allclose(track.camera(77).centre, base.centre, atol=1e-6)
