"""Draw people tracks: a colour and number per track, a dot at the feet, and a top-down court
map in the corner with everyone placed at their court position (from the feet)."""

from __future__ import annotations

import cv2
import numpy as np

from setter_eyes.track.people import PeopleTracks


def track_colour(tid: int) -> tuple[int, int, int]:
    hue = (tid * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class TracksOverlay:
    def __init__(self, tracks: PeopleTracks):
        self.t = tracks
        self.frames = np.unique(tracks.frame)
        order = np.argsort(tracks.frame, kind="stable")
        self.order = order
        self.starts = np.searchsorted(tracks.frame[order], self.frames)
        self.ends = np.searchsorted(tracks.frame[order], self.frames, side="right")

    def rows_at(self, frame: int) -> np.ndarray:
        """Rows of the latest processed frame at or before `frame`."""
        k = int(np.searchsorted(self.frames, frame, side="right")) - 1
        if k < 0:
            return np.zeros(0, int)
        return self.order[self.starts[k]:self.ends[k]]

    def draw(self, image: np.ndarray, frame: int, s: float) -> int:
        rows = self.rows_at(frame)
        th = max(1, round(image.shape[1] / 960))
        for r in rows:
            tid = int(self.t.track_id[r])
            c = track_colour(tid)
            x1, y1, x2, y2 = (self.t.box[r] * s).round().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), c, th)
            cv2.circle(image, ((x1 + x2) // 2, y2), 3 * th, c, -1, cv2.LINE_AA)
            cv2.putText(image, str(tid), (x1, y1 - 4 * th), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * th, c, th, cv2.LINE_AA)
        self._court_map(image, rows)
        return len(rows)

    def _court_map(self, image: np.ndarray, rows: np.ndarray) -> None:
        """Top-down court in the top-right corner: far end at the top, our side at the bottom."""
        h, w = image.shape[:2]
        px = h / 40                                  # pixels per metre
        margin = 3.0
        x_lo, x_hi, y_lo, y_hi = -margin, 9 + margin, -9 - margin, 9 + margin
        mw, mh = int((x_hi - x_lo) * px), int((y_hi - y_lo) * px)
        ox, oy = w - mw - 10, 40
        panel = image[oy:oy + mh, ox:ox + mw]
        panel[:] = (panel * 0.35).astype(np.uint8)

        def to_px(x, y):
            return int((x - x_lo) * px) + ox, int((y - y_lo) * px) + oy

        cv2.rectangle(image, to_px(0, -9), to_px(9, 9), (0, 255, 255), 1)
        for y in (-3, 3):
            cv2.line(image, to_px(0, y), to_px(9, y), (0, 200, 200), 1)
        cv2.line(image, to_px(-0.5, 0), to_px(9.5, 0), (255, 0, 255), 2)      # net
        for r in rows:
            xy = self.t.court_xy_smooth[r]
            if not np.isfinite(xy).all() or not (x_lo <= xy[0] <= x_hi and y_lo <= xy[1] <= y_hi):
                continue
            tid = int(self.t.track_id[r])
            cv2.circle(image, to_px(*xy), max(3, int(px * 0.25)), track_colour(tid), -1,
                       cv2.LINE_AA)
