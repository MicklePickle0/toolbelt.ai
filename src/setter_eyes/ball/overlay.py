"""Draw the tracked ball on video frames.

The ball draws its own trail: during a rally, a line behind the ball from its last touch
(it starts fresh at every touch), and an X with its action at the latest touch, replaced by
the next (yellow = us, blue = them, red = floor). Between rallies, a short trail. Trail
points are
kept in the pan-free reference view and redrawn through the current camera, so the line
stays on the ball's real path while the camera pans. Optional: raw detections, and
(--ground) a drop line from the ball to the floor with its ground trail, from the fitted
3D flight (CLAUDE.md step 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath, Detections, unstabilise
from setter_eyes.calibration.camera import Camera

COLOURS = {"detection": (160, 160, 160), "measured": (0, 255, 0), "bridged": (0, 165, 255),
           "trail": (255, 255, 255), "ground": (120, 255, 120),
           "us": (0, 220, 255), "them": (255, 140, 60), "floor": (0, 0, 255),
           None: (200, 200, 200)}


class BallOverlay:
    def __init__(self, cache_dir: Path, min_confidence: float, ref: Camera,
                 trail_s: float = 1.0, show_detections: bool = False,
                 trail_px: float = 3.0):
        self.path = BallPath.load(cache_dir / "ball_path.npz")
        self.dets = Detections.load(cache_dir / "ball_detections.npz")
        d = json.loads((cache_dir / "ball_arcs.json").read_text(encoding="utf-8"))
        self.fps = float(d["fps"])
        self.g = np.array([0.0, 0.0, -float(d["gravity"])])
        self.arcs = [Arc(**a) for a in d["arcs"]]
        self.min_conf = min_confidence
        self.ref = ref
        step = max(1, float(np.median(np.diff(self.path.frames))))
        self.trail = max(1, round(trail_s * self.fps / step))
        self.groups = self.dets.by_frame()
        self.show_detections = show_detections
        self.trail_px = trail_px
        self.rallies, self.contacts = [], []
        if (cache_dir / "rallies.json").exists() and (cache_dir / "contacts.json").exists():
            self.rallies = json.loads((cache_dir / "rallies.json").read_text())["rallies"]
            self.contacts = json.loads((cache_dir / "contacts.json").read_text())["contacts"]

    def _index(self, frame: int) -> int:
        return int(np.searchsorted(self.path.frames, frame, side="right")) - 1

    def arc_at(self, k: int) -> Arc | None:
        for arc in self.arcs:
            if arc.first <= k <= arc.last:
                return arc
        return None

    def rally_at(self, t: float) -> dict | None:
        for ra in self.rallies:
            if ra["start_t"] <= t <= ra["end_t"] + 1.0:
                return ra
        return None

    def _to_view(self, uv_ref: np.ndarray, cam: Camera, s: float) -> np.ndarray:
        return unstabilise(cam, self.ref, uv_ref) * s

    def draw(self, image: np.ndarray, frame: int, cam: Camera, s: float,
             ground: bool = False) -> str:
        """Draw onto image (scaled by s from the full-resolution cam) -> a status label."""
        k = self._index(frame)
        if k < 0:
            return ""
        th = max(1, round(image.shape[1] / 960))
        t_now = frame / self.fps
        if self.show_detections:
            for i in self.groups.get(int(self.path.frames[k]), []):
                x1, y1, x2, y2 = (self.dets.det_box[i] * s).round().astype(int)
                cv2.rectangle(image, (x1, y1), (x2, y2), COLOURS["detection"], 1)

        rally = self.rally_at(t_now)
        if rally is not None:
            # The line starts at the last touch (or the serve / rally start) before now.
            lo = int(np.searchsorted(self.path.frames, rally["start_frame"]))
            for i in rally["contacts"]:
                c = self.contacts[i]
                if c["t"] <= t_now:
                    lo = max(lo, c["path_index"])
        else:
            lo = max(0, k - self.trail)
        self._trail(image, lo, k, cam, s, th)
        if rally is not None:
            # Only the latest touch keeps its X; the next touch replaces it.
            past = [self.contacts[i] for i in rally["contacts"]
                    if self.contacts[i]["t"] <= t_now]
            if past:
                self._x(image, past[-1], cam, s, th)

        label = ""
        self.ball_xyz = None
        arc = self.arc_at(k)
        if arc is not None and arc.confidence >= self.min_conf:
            now = min(max(t_now, arc.t0), arc.t0 + arc.duration_s)
            p = arc.position(np.array([now]), self.g)[0]
            v = np.array(arc.v0) + self.g * (now - arc.t0)
            label = f"ball z {p[2]:.1f} m  {np.linalg.norm(v):.0f} m/s"
            self.ball_xyz = p
            if ground:
                self._ground(image, arc, p, now, cam.scaled(s), th)
        if np.isfinite(self.path.uv[k]).all():
            u, v = (self.path.uv[k] * s).round().astype(int)
            r = max(6, round(float(np.nan_to_num(self.path.size_px[k], nan=30.0)) * s * 0.7))
            cv2.circle(image, (u, v), r, COLOURS["measured" if self.path.measured[k]
                                              else "bridged"], th + 1, cv2.LINE_AA)
        return label

    def _trail(self, image, lo: int, k: int, cam: Camera, s: float, th: int) -> None:
        """The ball's path from sample lo to k, split where the ball was lost or the tracker
        switched to another ball (so no straight jumps are drawn)."""
        ok = np.isfinite(self.path.uv_ref[lo:k + 1]).all(axis=1)
        tid = self.path.track_id[lo:k + 1]
        runs, run = [], []
        for j in range(lo, k + 1):
            if not ok[j - lo] or (run and tid[j - lo] != tid[run[-1] - lo]):
                if len(run) > 1:
                    runs.append(run)
                run = [j] if ok[j - lo] else []
            else:
                run.append(j)
        if len(run) > 1:
            runs.append(run)
        for r in runs:
            pts = self._to_view(self.path.uv_ref[r], cam, s)
            width = max(1, round(self.trail_px * image.shape[1] / 1920))
            cv2.polylines(image, [pts.round().astype(np.int32)], False, COLOURS["trail"],
                          width, cv2.LINE_AA)

    def _x(self, image, c: dict, cam: Camera, s: float, th: int) -> None:
        """An X where the ball changed direction: yellow = us, blue = them, red = floor."""
        k = c["path_index"]
        near = [j for j in range(max(0, k - 10), min(len(self.path.frames), k + 11))
                if np.isfinite(self.path.uv_ref[j]).all()]
        if not near:
            return
        uv_ref = self.path.uv_ref[min(near, key=lambda j: abs(j - k))]
        u, v = self._to_view(uv_ref[None], cam, s)[0].round().astype(int)
        colour = COLOURS["floor"] if c["kind"] == "floor" else COLOURS[c["team"]]
        r = 9 * th
        for (a, b), (cc, w) in ((((u - r, v - r), (u + r, v + r)), ((0, 0, 0), 5 * th)),
                                (((u - r, v + r), (u + r, v - r)), ((0, 0, 0), 5 * th)),
                                (((u - r, v - r), (u + r, v + r)), (colour, 2 * th)),
                                (((u - r, v + r), (u + r, v - r)), (colour, 2 * th))):
            cv2.line(image, a, b, cc, w, cv2.LINE_AA)
        if c.get("action"):                  # the touch's action (classify/rules.py)
            text = c["action"].replace("_", " ").upper()
            org = (u + r + 4 * th, v + 4 * th)
            scale = 0.45 * th
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4 * th,
                        cv2.LINE_AA)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, th + 1,
                        cv2.LINE_AA)

    def _ground(self, image, arc: Arc, p, now: float, cam: Camera, th: int) -> None:
        floor = np.array([p[0], p[1], 0.0])
        (bu, fu), zz = cam.project(np.stack([p, floor]))
        if (zz > 0.1).all():
            cv2.line(image, tuple(bu.round().astype(int)), tuple(fu.round().astype(int)),
                     COLOURS["ground"], th, cv2.LINE_AA)
        past = arc.position(np.linspace(arc.t0, now, 30), self.g) * [1.0, 1.0, 0.0]
        guv, gz = cam.project(past)
        guv = guv[gz > 0.1]
        if len(guv) > 1:
            cv2.polylines(image, [guv.round().astype(np.int32)], False, COLOURS["ground"], th,
                          cv2.LINE_AA)
