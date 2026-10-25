"""Generate a synthetic labelled workspace to smoke-test the label tooling end to end.

Writes two short 1280x720 clips (a drawn court and a ball flying on arcs), with
a manifest, events.csv and boxes.coco.json in the real formats. Not training
data: it exists to exercise ingest -> validate -> import -> preview -> export.

Usage:
  python scripts/make_demo_data.py [--out data/demo]
  python scripts/ingest_labels.py --manifest data/demo/manifest.yaml --db data/demo/labels.db
  python scripts/preview_labels.py --manifest data/demo/manifest.yaml --video demo_a
  python scripts/build_detection_dataset.py --manifest data/demo/manifest.yaml \
      --out data/demo/detection
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from setter_eyes import config

W, H, FPS = 1280, 720, 30.0
# Contacts per rally as (frame offset, team, jersey, action, pass_score); the ball
# flies on a parabola between consecutive contacts.
RALLY = [(0, "them", None, "serve", None), (30, "us", 7, "receive", 3),
         (52, "us", 1, "set", None), (72, "us", 7, "attack", None)]
RALLY_LEN = 100


def _court(img: np.ndarray) -> None:
    img[:] = (60, 110, 160)
    pts = np.array([[140, 700], [1140, 700], [900, 330], [380, 330]], np.int32)
    cv2.fillPoly(img, [pts], (70, 140, 200))
    cv2.polylines(img, [pts], True, (255, 255, 255), 3)
    cv2.line(img, (330, 460), (950, 460), (255, 255, 255), 2)       # attack line
    cv2.line(img, (360, 330), (920, 330), (230, 230, 230), 2)       # net base
    cv2.rectangle(img, (360, 200), (920, 210), (240, 240, 240), -1)  # net tape


def _ball_at(f: int, anchors: list[tuple[int, tuple[float, float]]]) -> tuple[float, float]:
    for (f0, p0), (f1, p1) in zip(anchors, anchors[1:], strict=False):
        if f0 <= f <= f1:
            s = (f - f0) / (f1 - f0)
            x = p0[0] + s * (p1[0] - p0[0])
            y = p0[1] + s * (p1[1] - p0[1]) - 4 * 180 * s * (1 - s)  # arc apex 180 px up
            return x, y
    return anchors[-1][1]


def make_clip(out: Path, key: str, n_rallies: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    video_path = out / "videos" / f"{key}.mp4"
    label_dir = out / "labels" / key
    video_path.parent.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []
    ball: dict[int, tuple[float, float]] = {}
    start = 15
    for r in range(n_rallies):
        contacts = [(start + off, *rest) for off, *rest in RALLY]
        end = start + RALLY_LEN
        spots = [(640 + rng.uniform(-150, 150), 260)] + \
                [(rng.uniform(400, 880), rng.uniform(380, 620)) for _ in contacts[1:]] + \
                [(rng.uniform(420, 860), 250)]
        anchors = [(c[0], p) for c, p in zip(contacts, spots, strict=False)]
        anchors.append((end - 5, spots[-1]))
        for f in range(start, end - 5):
            ball[f] = _ball_at(f, anchors)
        events.append({"set": 1, "frame": start - 5, "type": "rally_start"})
        for f, team, jersey, action, pass_score in contacts:
            events.append({"set": 1, "frame": f, "type": "contact", "team": team,
                           "jersey": jersey, "action": action, "pass_score": pass_score})
        winner = "us" if r % 2 == 0 else "them"
        events.append({"set": 1, "frame": end, "type": "rally_end", "point_winner": winner,
                       "outcome": "kill" if winner == "us" else "attack_error"})
        start = end + 30
    n_frames = start

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    base = np.zeros((H, W, 3), np.uint8)
    _court(base)
    for f in range(n_frames):
        img = base.copy()
        if f in ball:
            cv2.circle(img, (round(ball[f][0]), round(ball[f][1])), 9, (40, 220, 250), -1)
        writer.write(img)
    writer.release()

    fields = ["set", "frame", "t", "type", "team", "jersey", "action", "pass_score",
              "point_winner", "outcome", "note"]
    with (label_dir / "events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for e in events:
            w.writerow({k: ("" if v is None else v) for k, v in e.items()})

    # Box labels on every 5th frame, like a sparse human labelling pass (negatives included).
    images, anns = [], []
    for i, f in enumerate(range(0, n_frames, 5), start=1):
        images.append({"id": i, "file_name": f"frame_{f:06d}.png", "width": W, "height": H})
        if f in ball:
            x, y = ball[f]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1,
                         "bbox": [x - 10, y - 10, 20, 20], "area": 400, "iscrowd": 0})
    (label_dir / "boxes.coco.json").write_text(json.dumps(
        {"images": images, "annotations": anns,
         "categories": [{"id": 1, "name": "ball", "supercategory": "none"}]}), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.repo_path("data/demo"))
    args = ap.parse_args()

    clips = {"demo_a": ("train", 3, "Demo Gym A"), "demo_b": ("valid", 2, "Demo Gym B")}
    for i, (key, (_, n, _)) in enumerate(clips.items()):
        make_clip(args.out, key, n, seed=i)
    lines = ["team: Demo Team", "roster:"]
    lines += [f"  - {{number: {n}, name: Player {n}}}" for n in (1, 2, 3, 4, 5, 7)]
    lines.append("videos:")
    for key, (split, _, venue) in clips.items():
        lines += [f"  - id: {key}", f"    path: videos/{key}.mp4", f"    venue: {venue}",
                  "    net_height: boys_17_18u",
                  "    ball: mikasa_v200w_yellow", f"    split: {split}",
                  "    labels:", f"      events: labels/{key}/events.csv",
                  f"      boxes: labels/{key}/boxes.coco.json",
                  "    lineups:", "      1: [1, 7, 2, 3, 4, 5]"]
    (args.out / "manifest.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote demo workspace to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
