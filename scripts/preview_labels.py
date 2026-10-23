"""Render labels over the video to check them by eye (visual verification).

Draws labelled boxes, the labelled ball trail within the rally, and a HUD with
set/rally, the contact chain so far and the point winner.

Usage:
  python scripts/preview_labels.py --video KEY                     # whole video -> mp4
  python scripts/preview_labels.py --video KEY --rally 1:4         # set 1, rally 4 only
  python scripts/preview_labels.py --video KEY --start 60 --end 90 --scale 0.5
  python scripts/preview_labels.py --video KEY --contacts          # one JPEG per contact
Output goes to <output_dir>/ (data/outputs by default).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from setter_eyes import config
from setter_eyes.ingest import iter_frames, read_frames, write_image
from setter_eyes.labels import RallyLabel, load_manifest, load_video_labels

COLOURS = {"ball": (0, 220, 255), "person": (255, 160, 0)}   # BGR
TEAM_COLOURS = {"us": (80, 220, 80), "them": (80, 80, 240)}
BOX_HOLD_FRAMES = 3        # keep a sparse label visible for a few frames
CONTACT_FLASH_S = 0.7
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest video key")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--rally", help="SET:RALLY, e.g. 1:4")
    ap.add_argument("--start", type=float, help="seconds")
    ap.add_argument("--end", type=float, help="seconds")
    ap.add_argument("--scale", type=float, default=1.0, help="output resize factor")
    ap.add_argument("--contacts", action="store_true", help="write one still per contact")
    ap.add_argument("--out", type=Path, help="output file/dir")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    vl = load_video_labels(manifest, manifest.video(args.video))
    for issue in vl.issues:
        print(issue)
    if vl.info is None:
        return 1
    info, fps = vl.info, vl.info.fps
    out_dir = config.repo_path(paths["output_dir"])

    start = round(args.start * fps) if args.start is not None else 0
    end = round(args.end * fps) if args.end is not None else None
    rallies = vl.rallies
    if args.rally:
        s, r = (int(x) for x in args.rally.split(":"))
        rallies = [x for x in rallies if (x.set_number, x.rally_number) == (s, r)]
        if not rallies:
            print(f"no rally {args.rally}")
            return 1
        start, end = max(0, rallies[0].start_frame - round(fps)), rallies[0].end_frame + round(fps)

    boxes = vl.boxes.frames if vl.boxes else {}

    if args.contacts:
        target = args.out or out_dir / f"{args.video}_contacts"
        wanted = {c.frame: (r, c) for r in rallies for c in r.contacts
                  if start <= c.frame and (end is None or c.frame < end)}
        for frame in read_frames(info.path, wanted):
            rally, _ = wanted[frame.index]
            img = draw(frame.image, frame.index, fps, rally, boxes)
            img = _resize(img, args.scale)
            c = wanted[frame.index][1]
            name = (f"s{rally.set_number}r{rally.rally_number:03d}_f{frame.index:06d}_"
                    f"{c.team}_{c.action or 'unknown'}.jpg")
            write_image(target / name, img)
        print(f"wrote {len(wanted)} stills to {target}")
        return 0

    target = args.out or out_dir / f"{args.video}_labels.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    size = (round(info.width * args.scale), round(info.height * args.scale))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = 0
    try:
        for frame in iter_frames(info.path, start=start, end=end):
            rally = next((r for r in rallies if r.start_frame <= frame.index <= r.end_frame), None)
            writer.write(_resize(draw(frame.image, frame.index, fps, rally, boxes), args.scale))
            n += 1
    finally:
        writer.release()
    print(f"wrote {n} frames to {target}")
    return 0


def draw(image: np.ndarray, index: int, fps: float, rally: RallyLabel | None,
         boxes: dict) -> np.ndarray:
    img = image.copy()
    for back in range(BOX_HOLD_FRAMES):
        for b in boxes.get(index - back, []):
            colour = COLOURS.get(b.cls, (255, 255, 255))
            p1, p2 = (round(b.x), round(b.y)), (round(b.x + b.w), round(b.y + b.h))
            cv2.rectangle(img, p1, p2, colour, 2 if back == 0 else 1)
            if back == 0:
                _text(img, b.cls, (p1[0], p1[1] - 4), 0.5, colour)

    white = (255, 255, 255)
    lines = [(f"frame {index}  t {index / fps:7.2f}s", white)]
    if rally is not None:
        trail = [b.centre for f in sorted(boxes) if rally.start_frame <= f <= index
                 for b in boxes[f] if b.cls == "ball"]
        if len(trail) > 1:
            pts = np.array(trail, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, COLOURS["ball"], 2)
        lines.append((f"set {rally.set_number}  rally {rally.rally_number}  "
                      f"serve: {rally.serving_team or '?'}", white))
        for c in rally.contacts:
            if c.frame > index:
                break
            flash = (index - c.frame) / fps < CONTACT_FLASH_S
            who = f"#{c.jersey}" if c.jersey is not None else c.team
            extra = f"  pass {c.pass_score}" if c.pass_score is not None else ""
            lines.append((f"{'>' if flash else ' '} {c.frame:6d} {who:5} {c.action or '?'}{extra}",
                          TEAM_COLOURS[c.team]))
        if index >= rally.end_frame:
            lines.append((f"POINT {str(rally.point_winner).upper()}  ({rally.outcome or '-'})",
                          TEAM_COLOURS.get(rally.point_winner or "", white)))

    width = max(cv2.getTextSize(line, FONT, 0.7, 2)[0][0] for line, _ in lines)
    panel = img[:20 + 26 * len(lines), :min(img.shape[1], width + 24)]
    panel[:] = (panel * 0.35).astype(np.uint8)  # darken behind the HUD
    for i, (line, colour) in enumerate(lines):
        _text(img, line, (12, 28 + 26 * i), 0.7, colour)
    return img


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
          colour: tuple[int, int, int]) -> None:
    cv2.putText(img, text, org, FONT, scale, colour, 2, cv2.LINE_AA)


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    raise SystemExit(main())
