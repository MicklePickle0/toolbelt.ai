"""Highlight reels from search results (owner, 2026-11-01; Hudl style).

Each picked touch becomes a clip from `before` to `after` seconds around it, with a ring
around the feet of the player who made it (track/spotlight.py `draw_ring`): the player
whose hands were at the ball (identity/toucher.py, when the video is posed), else the
near-side person nearest the ball, followed through the clip (`follow`). Clips from any
sets are re-encoded alike (`width` wide, `fps`) and joined into one MP4.

Reels are made one at a time on a background thread; `status` reports progress and where
the file is (data/outputs/reels/<id>_<name>.mp4).
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath
from setter_eyes.calibration.track import PoseTrack
from setter_eyes.identity.toucher import Poses, TouchPicker
from setter_eyes.ingest import probe
from setter_eyes.track.people import PeopleTracks
from setter_eyes.track.spotlight import ball_xyz, draw_ring, follow, position_at, toucher_track


class _Video:
    """What a video's clips need, loaded once per reel."""

    def __init__(self, entry, cache: Path, calibration_dir: Path, cfg: dict):
        proxy = cache / "proxy.mp4"
        self.src = proxy if proxy.exists() else entry.path
        self.fps = probe(entry.path).fps
        self.ring = None
        track = calibration_dir / f"{entry.key}_track.npz"
        if (cache / "people_tracks.npz").exists() and track.exists():
            d = json.loads((cache / "ball_arcs.json").read_text(encoding="utf-8"))
            arcs = [Arc(**a) for a in d["arcs"]]
            g = np.array([0.0, 0.0, -float(d["gravity"])])
            poses = cache / "pose.npz"
            self.ring = {
                "cams": PoseTrack.load(track), "people": PeopleTracks.load(cache /
                                                                           "people_tracks.npz"),
                "path": BallPath.load(cache / "ball_path.npz"), "arcs": arcs, "g": g,
                "picker": TouchPicker(Poses.load(poses), self.fps, cfg["identity"]["toucher"])
                if poses.exists() else None}

    def toucher(self, frame: int) -> int | None:
        r = self.ring
        if r["picker"] is not None:
            k = int(np.argmin(np.abs(r["path"].frames - frame)))
            pick = r["picker"].pick(frame, r["path"].uv[k])
            if pick is not None:
                return pick[0]
        return toucher_track(r["path"], r["people"], frame,
                             xyz=ball_xyz(r["arcs"], r["g"], frame / self.fps))


def write_clip(v: _Video, out: Path, t0: float, t1: float, touch_t: float | None,
               width: int, fps: float) -> bool:
    """One clip, the toucher ringed when found. True when the ring was drawn."""
    f0, f1 = max(0, round(t0 * v.fps)), round(t1 * v.fps)
    rows, cams, scale = None, None, 1.0
    if touch_t is not None and v.ring is not None:
        tid = v.toucher(round(touch_t * v.fps))
        if tid is not None:
            rows = follow(v.ring["people"], tid, f0, f1, v.fps)
            cams = v.ring["cams"]
    cap = cv2.VideoCapture(str(v.src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, img = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"can't read {v.src.name} at {t0:.1f} s")
    h, w = img.shape[:2]
    if cams is not None:
        scale = w / cams.base.width
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{v.fps:.6f}", "-i", "-", "-vf", f"scale={width}:-2,fps={fps}", "-c:v",
         "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
         str(out)], stdin=subprocess.PIPE)
    f = f0
    while ok and f < f1:
        if rows is not None:
            pos = position_at(v.ring["people"], rows, f)
            if pos is not None:
                draw_ring(img, cams.camera(f), pos[0], pos[1], v.ring_cfg, scale)
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
        ok, img = cap.read()
        f += 1
    enc.stdin.close()
    enc.wait()
    cap.release()
    return rows is not None


class ReelMaker:
    def __init__(self, manifest, cache_dir: Path, calibration_dir: Path, out_dir: Path,
                 cfg: dict):
        """manifest: a callable returning the current Manifest."""
        self.manifest, self.cache_dir, self.calibration_dir = manifest, cache_dir, calibration_dir
        self.out_dir, self.cfg = out_dir, cfg
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._queue: list[tuple[str, dict]] = []
        self._wake = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, req: dict) -> str:
        rid = uuid.uuid4().hex[:8]
        with self._lock:
            self._jobs[rid] = {"id": rid, "state": "queued", "done": 0,
                               "total": len(req["items"]), "ringed": 0, "error": None,
                               "path": None, "created": time.time()}
            self._queue.append((rid, req))
        self._wake.set()
        return rid

    def status(self, rid: str) -> dict | None:
        with self._lock:
            st = self._jobs.get(rid)
            return dict(st) if st else None

    def _set(self, rid: str, **kw) -> None:
        with self._lock:
            self._jobs[rid].update(kw)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                job = self._queue.pop(0) if self._queue else None
                if not self._queue:
                    self._wake.clear()
            if job is None:
                continue
            rid, req = job
            try:
                self._make(rid, req)
            except Exception as e:                  # noqa: BLE001 - reported to the page
                self._set(rid, state="failed", error=str(e))

    def _make(self, rid: str, req: dict) -> None:
        self._set(rid, state="running")
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", req.get("name") or "reel").strip("_") or "reel"
        work = self.out_dir / f"{rid}_{name}"
        videos: dict[str, _Video] = {}
        ring_cfg = self.cfg["overlay"]["spotlight"]
        clips = []
        for n, it in enumerate(sorted(req["items"], key=lambda x: (x["video"], x["t"])), start=1):
            if it["video"] not in videos:
                v = _Video(self.manifest().video(it["video"]), self.cache_dir / it["video"],
                           self.calibration_dir, self.cfg)
                v.ring_cfg = ring_cfg
                videos[it["video"]] = v
            v = videos[it["video"]]
            out = work / f"{n:03d}.mp4"
            ringed = write_clip(v, out, it["t"] - req["before"], it["t"] + req["after"],
                                it["t"] if req.get("ring", True) else None, 1280, 30)
            clips.append(out)
            with self._lock:
                st = self._jobs[rid]
                st["done"] += 1
                st["ringed"] += int(ringed)
        listing = work / "list.txt"
        listing.write_text("".join(f"file '{c.name}'\n" for c in clips), encoding="utf-8")
        final = self.out_dir / f"{rid}_{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                        str(listing), "-c", "copy", str(final)], check=True)
        self._set(rid, state="done", path=str(final))
