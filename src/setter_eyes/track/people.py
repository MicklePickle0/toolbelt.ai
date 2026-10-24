"""People tracking and court positions (Milestone 2, steps 2-3).

Tracking: ByteTrack (Roboflow `trackers`, Apache-2.0) on boxes re-expressed in the
calibrated reference view, so the camera's pan doesn't look like everyone moving and
tracks survive it. ByteTrack's second pass uses low-score boxes to hold a track through
partial occlusion, which is why detections down to 0.3 are kept.

Court position: the bottom centre of a box (the feet) traced to the floor (z = 0) through
that frame's camera, in court metres; smoothed per track with a short rolling median.
Depth is less certain than lateral position (CLAUDE.md), and boxes whose bottom edge is
at the frame edge (feet cut off) are flagged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from setter_eyes.calibration.camera import MIRROR, Camera


def reference_view(cam: Camera, ref: Camera, uv: np.ndarray) -> np.ndarray:
    """Pixels seen by cam -> pixels the reference camera would see (same centre; exact for a
    camera that only rotates). Vectorised over points."""
    d = cam.rays(np.asarray(uv, float).reshape(-1, 2))          # court-frame directions
    pc = (d * MIRROR) @ ref.R.T
    return ref.f * pc[:, :2] / pc[:, 2:3] + ref.principal_point


def boxes_to_reference(cam: Camera, ref: Camera, boxes: np.ndarray) -> np.ndarray:
    """(N, 4) x1, y1, x2, y2 -> the axis-aligned box around their corners in the reference view."""
    if not len(boxes):
        return boxes.reshape(0, 4)
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                        np.stack([x1, y2], 1), np.stack([x2, y2], 1)], 1)   # (N, 4, 2)
    ref_pts = reference_view(cam, ref, corners.reshape(-1, 2)).reshape(-1, 4, 2)
    return np.concatenate([ref_pts.min(axis=1), ref_pts.max(axis=1)], axis=1)


def feet_on_floor(cam: Camera, boxes: np.ndarray) -> np.ndarray:
    """(N, 4) boxes -> (N, 2) court x, y (metres) where the bottom-centre ray meets the floor.
    NaN if the ray doesn't go down to the floor."""
    if not len(boxes):
        return np.zeros((0, 2))
    feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    d = cam.rays(feet)
    c = cam.centre
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -c[2] / d[:, 2]
    xy = c[:2] + lam[:, None] * d[:, :2]
    xy[~(lam > 0)] = np.nan
    return xy


@dataclass
class PeopleTracks:
    """One row per tracked detection."""
    frame: np.ndarray
    track_id: np.ndarray
    box: np.ndarray            # (N, 4) image pixels of that frame
    score: np.ndarray
    court_xy: np.ndarray       # (N, 2) metres, from the feet
    court_xy_smooth: np.ndarray
    feet_visible: np.ndarray   # bool: box bottom not at the frame edge

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> PeopleTracks:
        d = np.load(path)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def summary(self, cfg: dict) -> list[dict]:
        """Per track: span, length and where on court it spends its time (for step 4)."""
        out = []
        for tid in np.unique(self.track_id):
            sel = self.track_id == tid
            xy = self.court_xy_smooth[sel]
            ok = np.isfinite(xy).all(axis=1) & self.feet_visible[sel]
            ours = ok & (xy[:, 1] >= -cfg["our_half_margin_m"]) & \
                (xy[:, 1] <= 9 + cfg["our_half_margin_m"]) & \
                (xy[:, 0] >= -cfg["our_half_margin_m"]) & (xy[:, 0] <= 9 + cfg["our_half_margin_m"])
            med = np.median(xy[ok], axis=0) if ok.any() else np.array([np.nan, np.nan])
            out.append({"track_id": int(tid), "first_frame": int(self.frame[sel].min()),
                        "last_frame": int(self.frame[sel].max()), "samples": int(sel.sum()),
                        "mean_score": round(float(self.score[sel].mean()), 3),
                        "median_court_xy": [round(float(v), 2) for v in med],
                        "share_in_our_half": round(float(ours.sum() / max(1, ok.sum())), 3)})
        return out


def track_people(frames: np.ndarray, det_frame: np.ndarray, det_box: np.ndarray,
                 det_score: np.ndarray, fps: float, camera_at, ref: Camera, image_height: int,
                 cfg: dict) -> PeopleTracks:
    """frames: every processed frame (in order); camera_at(frame) -> that frame's Camera."""
    import supervision as sv
    from trackers import ByteTrackTracker

    stride = max(1, int(np.median(np.diff(frames)))) if len(frames) > 1 else 1
    bt = cfg["bytetrack"]
    tracker = ByteTrackTracker(
        lost_track_buffer=bt["lost_track_buffer"], frame_rate=fps / stride,
        track_activation_threshold=bt["track_activation_threshold"],
        minimum_consecutive_frames=bt["minimum_consecutive_frames"],
        minimum_iou_threshold=bt["minimum_iou_threshold"],
        high_conf_det_threshold=bt["high_conf_det_threshold"])
    order = np.argsort(det_frame, kind="stable")
    starts = np.searchsorted(det_frame[order], frames)
    ends = np.searchsorted(det_frame[order], frames, side="right")

    rows = {k: [] for k in ("frame", "track_id", "box", "score", "court_xy", "feet_visible")}
    for f, a, b in zip(frames, starts, ends, strict=True):
        idx = order[a:b]
        cam = camera_at(int(f))
        boxes, scores = det_box[idx].astype(float), det_score[idx].astype(float)
        dets = sv.Detections(xyxy=boxes_to_reference(cam, ref, boxes), confidence=scores,
                             class_id=np.zeros(len(idx), int), data={"i": np.arange(len(idx))})
        out = tracker.update(dets, timestamp=float(f) / fps)
        keep = out.tracker_id is not None and len(out) and out.tracker_id >= 0
        if not np.any(keep):
            continue
        out = out[out.tracker_id >= 0]
        i = out.data["i"]
        rows["frame"].append(np.full(len(i), f))
        rows["track_id"].append(out.tracker_id)
        rows["box"].append(boxes[i])
        rows["score"].append(scores[i])
        rows["court_xy"].append(feet_on_floor(cam, boxes[i]))
        rows["feet_visible"].append(boxes[i, 3] < image_height - cfg["feet_edge_px"])

    if not rows["frame"]:
        empty = np.zeros(0)
        return PeopleTracks(empty.astype(int), empty.astype(int), np.zeros((0, 4)), empty,
                            np.zeros((0, 2)), np.zeros((0, 2)), empty.astype(bool))
    cat = {k: np.concatenate(v) for k, v in rows.items()}
    smooth = cat["court_xy"].copy()
    w = cfg["smooth_samples"]
    for tid in np.unique(cat["track_id"]):
        sel = np.flatnonzero(cat["track_id"] == tid)
        xy = cat["court_xy"][sel]
        for j in range(len(sel)):
            win = xy[max(0, j - w // 2):j + w // 2 + 1]
            win = win[np.isfinite(win).all(axis=1)]
            if len(win):
                smooth[sel[j]] = np.median(win, axis=0)
    return PeopleTracks(cat["frame"], cat["track_id"], cat["box"], cat["score"],
                        cat["court_xy"], smooth, cat["feet_visible"])


def save_summary(tracks: PeopleTracks, cfg: dict, path: Path) -> list[dict]:
    summary = tracks.summary(cfg)
    path.write_text(json.dumps({"tracks": summary}, indent=1), encoding="utf-8")
    return summary
