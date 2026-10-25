import cv2
import numpy as np

from setter_eyes.calibration.transfer import transfer_calibration
from test_calibration import CFG, H, W, look_at

# A textured floor, x -4..13 m, y -14..14 m, at 60 px per metre.
PX = 60
X0, Y0 = -4.0, -14.0


def _floor_texture(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = np.full((28 * PX, 17 * PX, 3), (150, 190, 215), np.uint8)
    for _ in range(2500):
        c = tuple(int(v) for v in rng.integers(0, [17 * PX, 28 * PX]))
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(tex, c, int(rng.integers(3, 14)), colour, -1)
    return tex


def _render(cam, tex) -> np.ndarray:
    """The floor seen by cam: the floor-plane homography (no lens distortion)."""
    gx, gy = np.meshgrid(np.linspace(0, 17, 6), np.linspace(-14, 14, 8) - Y0)
    tpx = np.column_stack([gx.ravel(), gy.ravel()]) * PX
    pts = np.column_stack([tpx / PX + [X0, Y0], np.zeros(len(tpx))])
    uv, z = cam.project(pts)
    ok = z > 0.1
    Hm, _ = cv2.findHomography(tpx[ok].astype(np.float32), uv[ok].astype(np.float32))
    return cv2.warpPerspective(tex, Hm, (W, H), borderValue=(40, 40, 40))


def test_transfer_finds_a_moved_camera():
    tex = _floor_texture()
    before = look_at((4.5, 13.0, 1.6), (4.5, 0.0, 0.6))
    after = look_at((5.3, 13.4, 1.65), (3.8, 0.0, 0.5))       # moved and turned
    res = transfer_calibration(before, _render(before, tex), _render(after, tex), CFG)
    assert res is not None and res.confidence > 0.5
    np.testing.assert_allclose(res.camera.centre, after.centre, atol=0.05)
    probe = np.array([[0.0, 0.0, 0.0], [9.0, 9.0, 0.0], [4.5, 0.0, 2.43]])
    np.testing.assert_allclose(res.camera.project(probe)[0], after.project(probe)[0], atol=2.0)
