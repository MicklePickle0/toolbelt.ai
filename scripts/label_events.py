"""Label rallies and touches with the keyboard -> data/labels/<id>/events.csv.

Plays a seekable 1280-wide copy of the video (data/cache/<id>/proxy.mp4, made on first
use with ffmpeg; frame numbers match the original exactly) and writes events.csv as you
go. Opening an existing events.csv continues it. Rules for what to label:
docs/labelling.md.

Moving around:
  Space          play / pause            -              toggle slow motion (x0.25)
  Left / Right   one frame               ,  .           10 frames
  [  ]           1 second                {  }           5 seconds
  p  n           previous / next labelled event
  g              go to a time (type m:ss, Enter)
Labelling (at the frame shown):
  s              rally start (the server's toss)
  u / t          touch by us / them, then the action:
                 1 serve 2 receive 3 set 4 attack 5 block 6 dig 7 free_ball 8 dump
                 9 overpass 0 coverage;  ours: then jersey digits + Enter (Enter = unknown);
                 our receive: then pass score 0-4 (Enter skips)
  e              rally end, then winner (u / t), then outcome (Enter skips):
                 k kill  a attack_error  r reception_error  c ace  v service_error
                 b block_point  h ball_handling_error  o other
  z              undo the last label          x  delete the label on this frame
  Esc            cancel a half-entered label; otherwise quit (everything is saved)

Usage:
  python scripts/label_events.py --video titan-rise_set1 [--start 0:20]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from setter_eyes import config
from setter_eyes.ingest import probe
from setter_eyes.labels import load_manifest
from setter_eyes.labels.event_editor import ESC, EventEditor, clock
from setter_eyes.labels.events import parse_time

LEFT, UP, RIGHT, DOWN = 2424832, 2490368, 2555904, 2621440
WIDTH = 1280
HELP = ("Space play  <- -> frame  , . 10f  [ ] 1s  { } 5s  - slowmo  p/n prev/next label  "
        "g goto | s start  u/t touch  e end  z undo  x delete  Esc quit")


def make_proxy(src: Path, dst: Path) -> None:
    """A 1280-wide H.264 copy with a keyframe every 15 frames, so seeking is quick."""
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tail = ["-g", "15", "-bf", "0", "-fps_mode", "passthrough", "-an", str(dst)]
    attempts = [
        ["ffmpeg", "-v", "error", "-y", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
         "-i", str(src), "-vf", f"scale_cuda={WIDTH}:-2", "-c:v", "h264_nvenc",
         "-preset", "p4", "-cq", "23", *tail],
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"scale={WIDTH}:-2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", *tail],
    ]
    print(f"making a seekable copy of {src.name} (one-off, a few minutes)...")
    for cmd in attempts:
        if subprocess.run(cmd).returncode == 0:
            return
    raise SystemExit("could not make the proxy video")


class Player:
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = -1                      # index of the next frame read() returns, minus 1
        self.cache: dict[int, np.ndarray] = {}

    def frame(self, index: int) -> np.ndarray:
        index = int(np.clip(index, 0, self.count - 1))
        if index in self.cache:
            return self.cache[index]
        if index != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = self.cap.read()
        if not ok:
            img = np.zeros((720, WIDTH, 3), np.uint8)
        self.pos = index
        self.cache[index] = img
        if len(self.cache) > 600:          # keep ~10 s for stepping back
            for k in sorted(self.cache, key=lambda k: abs(k - index))[400:]:
                del self.cache[k]
        return img


def draw(img: np.ndarray, frame: int, fps: float, ed: EventEditor, playing: bool,
         slow: bool) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 64), (0, 0, 0), -1)
    state = ("PLAY x0.25" if slow else "PLAY") if playing else "PAUSED"
    cv2.putText(out, f"{clock(frame / fps)}   frame {frame}   {state}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    rally = ed.rally_at(frame)
    if rally is not None:
        touches = " > ".join(e.describe() for e in rally[1:] if e.frame <= frame)[-110:]
        cv2.putText(out, f"IN RALLY: {touches}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "between rallies", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 180, 180), 1, cv2.LINE_AA)
    here = ed.at(frame)
    if here:
        txt = "  |  ".join(e.describe() for e in here)
        cv2.putText(out, f"LABEL HERE: {txt}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, f"LABEL HERE: {txt}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)
    # Timeline: labels within 5 s either side.
    y0 = h - 70
    cv2.rectangle(out, (0, y0), (w, h), (0, 0, 0), -1)
    span = 5 * fps
    cv2.line(out, (w // 2, y0), (w // 2, y0 + 22), (255, 255, 255), 2)
    for e in ed.events:
        if abs(e.frame - frame) <= span:
            x = int(w / 2 + (e.frame - frame) / span * (w / 2 - 10))
            col = (0, 255, 0) if e.type != "contact" else \
                (0, 220, 255) if e.fields.get("team") == "us" else (255, 140, 60)
            cv2.line(out, (x, y0 + 2), (x, y0 + 20), col, 3)
    line2 = ed.prompt or ed.message
    cv2.putText(out, line2, (10, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255) if ed.prompt else (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(out, HELP, (10, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1,
                cv2.LINE_AA)
    return out


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--start", default="0", help="time to open at, e.g. 3:25")
    ap.add_argument("--set", type=int, default=1, help="set number written to each row")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    entry = manifest.video(args.video)
    info = probe(entry.path)
    fps = info.fps
    proxy = config.repo_path(paths["cache_dir"]) / entry.key / "proxy.mp4"
    if not proxy.exists():
        make_proxy(entry.path, proxy)
    out = entry.events or config.repo_path("data/labels") / entry.key / "events.csv"
    ed = EventEditor(out, fps, args.set)
    print(f"labels -> {out} ({len(ed.events)} already there)")

    player = Player(proxy)
    frame = round(parse_time(args.start) * fps)
    playing, slow, goto = False, False, None
    win = f"label events - {entry.key}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIDTH, 720)
    last = time.perf_counter()
    while True:
        cv2.imshow(win, draw(player.frame(frame), frame, fps, ed, playing, slow))
        if goto is not None:
            ed.message = f"go to (m:ss): {goto}_"
        k = cv2.waitKeyEx(1 if playing else 30)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        now = time.perf_counter()
        if playing:
            step = max(1, round((now - last) * fps * (0.25 if slow else 1.0)))
            frame = min(player.count - 1, frame + step)
        last = now
        if k == -1:
            continue
        ch = chr(k) if k < 256 else ""
        if goto is not None:                              # typing a time
            if ch == "\r":
                try:
                    frame = round(parse_time(goto) * fps)
                    ed.message = ""
                except ValueError:
                    ed.message = f"bad time {goto!r}"
                goto = None
            elif ch == ESC:
                goto, ed.message = None, ""
            elif ch == "\b":
                goto = goto[:-1]
            elif ch and (ch.isdigit() or ch in ":."):
                goto += ch
            continue
        if ed.pending is None:
            moves = {LEFT: -1, RIGHT: 1, ord(","): -10, ord("."): 10,
                     ord("["): -round(fps), ord("]"): round(fps),
                     ord("{"): -round(5 * fps), ord("}"): round(5 * fps)}
            if k in moves:
                playing = False
                frame = int(np.clip(frame + moves[k], 0, player.count - 1))
                continue
            if ch == " ":
                playing = not playing
                continue
            if ch == "-":
                slow = not slow
                continue
            if ch in ("p", "n"):
                nb = ed.neighbour(frame, ch == "n")
                if nb is not None:
                    frame, playing = nb, False
                continue
            if ch == "g":
                goto, playing = "", False
                continue
            if ch == ESC:
                break
        if ch and ed.key(ch, frame):
            playing = False
    cv2.destroyAllWindows()
    ed.save()
    print(f"saved {len(ed.events)} labels -> {out}\n"
          f"check: python scripts/ingest_labels.py --check --video {entry.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
