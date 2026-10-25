"""Draw people tracks: a colour and number per track, a dot at the feet, and a top-down court
map in the corner with everyone placed at their court position (from the feet). With roles
(track/roles.py): near-side players in colour, the far team thin grey, others faint;
filled (hidden-player) rows get a hollow feet dot."""

from __future__ import annotations

import cv2
import numpy as np

from setter_eyes.track.people import PeopleTracks


def track_colour(tid: int) -> tuple[int, int, int]:
    hue = (tid * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


GREY, FAINT = (170, 170, 170), (90, 90, 90)


def role_colour(tid: int, role: str) -> tuple[int, int, int]:
    return track_colour(tid) if role == "near_player" else GREY if role == "far_player" else FAINT


class TracksOverlay:
    def __init__(self, tracks: PeopleTracks, roles: dict[int, dict] | None = None):
        self.t = tracks
        self.roles = roles
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
        n_near = 0
        for r in rows:
            tid = int(self.t.track_id[r])
            role = self.role(tid)
            n_near += role == "near_player"
            c = role_colour(tid, role)
            w = 2 * th if role == "near_player" else 1
            x1, y1, x2, y2 = (self.t.box[r] * s).round().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), c, w)
            filled = int(self.t.source[r]) == 1
            cv2.circle(image, ((x1 + x2) // 2, y2), 3 * th, c, th if filled else -1, cv2.LINE_AA)
            if role == "near_player":
                cv2.putText(image, str(tid), (x1, y1 - 4 * th), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5 * th, c, th, cv2.LINE_AA)
        self._court_map(image, rows)
        return n_near if self.roles is not None else len(rows)

    def role(self, tid: int) -> str:
        if self.roles is None:
            return "near_player"
        return self.roles.get(tid, {}).get("role", "other")

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
            inside = x_lo <= xy[0] <= x_hi and y_lo <= xy[1] <= y_hi
            if not np.isfinite(xy).all() or not inside:
                continue
            tid = int(self.t.track_id[r])
            role = self.role(tid)
            c = role_colour(tid, role)
            size = int(px * 0.3) if role == "near_player" else int(px * 0.15)
            cv2.circle(image, to_px(*xy), max(2, size), c, -1, cv2.LINE_AA)


RING_GREEN = (60, 220, 60)


def draw_rings(image: np.ndarray, overlay: TracksOverlay, frame: int, cam, s: float,
               cfg: dict, in_rally: bool, radius_m: float = 0.45, alpha: float = 0.35) -> None:
    """A green ring on the floor around each of our players' feet (Hudl style): a circle of
    `radius_m` in court metres, projected through the camera so it lies on the court.
    Who gets one is decided per moment (roles.ours_now), not per whole track."""
    from setter_eyes.track.roles import ours_now

    rows = [r for r in overlay.rows_at(frame)
            if ours_now(overlay.t.court_xy_smooth[r], overlay.role(int(overlay.t.track_id[r])),
                        in_rally, cfg)]
    if not rows:
        return
    th = max(1, round(image.shape[1] / 960))
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    fill = image.copy()
    polys = []
    for r in rows:
        x, y = overlay.t.court_xy_smooth[r]
        if not np.isfinite([x, y]).all():
            continue
        circle = np.stack([x + radius_m * np.cos(a), y + radius_m * np.sin(a), np.zeros_like(a)],
                          axis=1)
        uv, z = cam.project(circle)
        if (z <= 0.1).any():
            continue
        poly = (uv * s).round().astype(np.int32)
        polys.append(poly)
        cv2.fillPoly(fill, [poly], RING_GREEN, cv2.LINE_AA)
    cv2.addWeighted(fill, alpha, image, 1 - alpha, 0, dst=image)
    for poly in polys:
        cv2.polylines(image, [poly], True, RING_GREEN, 2 * th, cv2.LINE_AA)


def draw_court_map(image: np.ndarray, overlay: TracksOverlay, frame: int, cfg: dict,
                   in_rally: bool, ball_xyz=None) -> None:
    """Top-down court in the top-right corner (far end at the top, our side at the bottom):
    our players green (same per-moment rule as the rings), the far team grey, the ball white
    when its 3D position is known."""
    from setter_eyes.track.roles import ours_now

    h, w = image.shape[:2]
    px = h / 40
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
    cv2.line(image, to_px(-0.5, 0), to_px(9.5, 0), (255, 0, 255), 2)
    for r in overlay.rows_at(frame):
        xy = overlay.t.court_xy_smooth[r]
        if not np.isfinite(xy).all() or not (x_lo <= xy[0] <= x_hi and y_lo <= xy[1] <= y_hi):
            continue
        role = overlay.role(int(overlay.t.track_id[r]))
        if ours_now(xy, role, in_rally, cfg):
            cv2.circle(image, to_px(*xy), max(3, int(px * 0.3)), RING_GREEN, -1, cv2.LINE_AA)
        elif role == "far_player" and xy[1] < 0.5:
            cv2.circle(image, to_px(*xy), max(2, int(px * 0.2)), GREY, -1, cv2.LINE_AA)
    if ball_xyz is not None and x_lo <= ball_xyz[0] <= x_hi and y_lo <= ball_xyz[1] <= y_hi:
        cv2.circle(image, to_px(ball_xyz[0], ball_xyz[1]), max(3, int(px * 0.22)),
                   (255, 255, 255), -1, cv2.LINE_AA)
