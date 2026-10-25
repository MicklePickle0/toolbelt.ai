import cv2
import numpy as np

from setter_eyes.calibration import Calibration
from setter_eyes.calibration.track import PoseTrack, reanchor
from test_calibration import look_at


def _yaw(deg: float) -> np.ndarray:
    return cv2.Rodrigues(np.array([0.0, np.radians(deg), 0.0]))[0]


def test_a_later_calibration_takes_over_after_the_camera_moved(tmp_path):
    before = look_at((4.5, 13.0, 1.6), (4.5, 0.0, 0.6))
    after = look_at((6.0, 13.8, 1.7), (4.5, 0.0, 0.6))     # tripod moved at frame 500
    frames = np.arange(0, 1000, 2)
    pan = 3 * np.sin(frames / 150)                          # degrees, the operator panning
    # The tracker measures the pan correctly but knows only the first calibration.
    main = np.array([cv2.Rodrigues(_yaw(a) @ before.R)[0].ravel() for a in pan])
    track = PoseTrack("v", before, 0, frames, main, np.ones(len(frames)))
    ref = 600                                               # recalibrated on frame 600
    k = int(np.searchsorted(frames, ref))
    later = Calibration("v", after, "manual", 1.0, 2.43, frame=ref, applies_from=500)

    fixed = reanchor(track, [later])
    for f, a in zip(frames, pan, strict=True):
        cam = fixed.camera(int(f))
        if f < 500:
            np.testing.assert_allclose(cam.R, _yaw(a) @ before.R, atol=1e-9)
            np.testing.assert_allclose(cam.centre, before.centre, atol=1e-9)
        else:
            np.testing.assert_allclose(cam.R, _yaw(a - pan[k]) @ after.R, atol=1e-9)
            np.testing.assert_allclose(cam.centre, after.centre, atol=1e-9)

    path = fixed.save(tmp_path / "t.npz")
    again = PoseTrack.load(path)
    assert [s.start for s in again.segments] == [500]
    np.testing.assert_allclose(again.camera(700).centre, after.centre, atol=1e-9)
    undone = reanchor(again, [])
    np.testing.assert_allclose(undone.rvecs, main)
    assert not undone.segments
