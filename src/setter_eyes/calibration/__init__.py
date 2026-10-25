"""Court calibration. Milestone 1.

Automatic line fitting with confidence, manual-click fallback, cv2.solvePnP camera pose,
drift check.
"""

from setter_eyes.calibration.auto import AutoResult, auto_calibrate
from setter_eyes.calibration.camera import Camera
from setter_eyes.calibration.court_model import CourtModel, court_model
from setter_eyes.calibration.fit import refine_on_lines, solve_from_points
from setter_eyes.calibration.lines import LineScorer, Support, line_mask
from setter_eyes.calibration.overlay import draw_court
from setter_eyes.calibration.store import (
    Calibration,
    load,
    load_segments,
    median_background,
    sample_frames,
    save,
)

__all__ = ["AutoResult", "Calibration", "Camera", "CourtModel", "LineScorer", "Support",
           "auto_calibrate", "court_model", "draw_court", "line_mask", "load", "load_segments",
           "median_background", "refine_on_lines", "sample_frames", "save",
           "solve_from_points"]
