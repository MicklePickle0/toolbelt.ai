"""Bounding-box labels from a COCO export (CVAT, Roboflow, Label Studio, ...).

Each COCO image must map back to a frame of the source video, either through
an integer "frame"/"frame_id" field or through its file name (patterns in
config/labels.yaml). Labelled images with no boxes are kept: they are the
negatives (e.g. frames where the ball is out of view).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from setter_eyes import config
from setter_eyes.labels.issues import Issues

_ROBOFLOW_SUFFIX = re.compile(r"_(?:jpe?g|png)\.rf\.[0-9a-fA-F]+$")


@dataclass(frozen=True)
class Box:
    cls: str          # canonical class name (config/labels.yaml box_classes)
    x: float          # top-left, px in the source frame
    y: float
    w: float
    h: float

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class BoxLabels:
    width: int | None = None     # labelled image size (should equal the video's)
    height: int | None = None
    frames: dict[int, list[Box]] = field(default_factory=dict)  # frame -> boxes (may be empty)

    def count(self, cls: str) -> int:
        return sum(b.cls == cls for boxes in self.frames.values() for b in boxes)


def frame_from_name(file_name: str, patterns: list[str] | None = None) -> int | None:
    stem = Path(file_name).stem
    stem = _ROBOFLOW_SUFFIX.sub("", stem)
    for pattern in patterns or config.load("labels")["frame_name_patterns"]:
        if m := re.search(pattern, stem, flags=re.IGNORECASE):
            return int(m.group(1))
    return None


def _class_lookup() -> dict[str, str]:
    lookup = {}
    for canonical, aliases in config.load("labels")["box_classes"].items():
        for name in [canonical, *aliases]:
            lookup[name.lower()] = canonical
    return lookup


def parse_coco(path: str | Path, frame_size: tuple[int, int] | None = None,
               frame_count: int | None = None) -> tuple[BoxLabels, Issues]:
    """Parse a COCO file into per-frame boxes. frame_size is the video's (width, height)."""
    path = Path(path)
    issues = Issues(str(path))
    labels = BoxLabels()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    lookup = _class_lookup()
    categories: dict[int, str | None] = {}
    for cat in data.get("categories", []):
        canonical = lookup.get(str(cat.get("name", "")).lower())
        categories[cat["id"]] = canonical
    unknown_cats = {c["name"] for c in data.get("categories", []) if not categories[c["id"]]}

    image_frame: dict[int, int] = {}
    sizes = set()
    for img in data.get("images", []):
        frame = img.get("frame", img.get("frame_id"))
        if frame is None:
            frame = frame_from_name(img.get("file_name", ""))
        if frame is None:
            issues.error(f"cannot get a frame index from image {img.get('file_name')!r} "
                         "(see frame_name_patterns in config/labels.yaml)")
            continue
        frame = int(frame)
        if frame in labels.frames:
            issues.error(f"frame {frame} appears in more than one image")
            continue
        if frame < 0 or (frame_count is not None and frame >= frame_count):
            issues.error(f"frame {frame} of {img.get('file_name')!r} is outside the video")
            continue
        image_frame[img["id"]] = frame
        labels.frames[frame] = []
        sizes.add((img.get("width"), img.get("height")))

    if len(sizes) > 1:
        issues.error(f"images have different sizes: {sorted(sizes)}")
    elif sizes:
        labels.width, labels.height = sizes.pop()
        if frame_size and (labels.width, labels.height) != tuple(frame_size):
            issues.error(f"labelled images are {labels.width}x{labels.height} but the video is "
                         f"{frame_size[0]}x{frame_size[1]}; label the original resolution")

    skipped = 0
    for ann in data.get("annotations", []):
        frame = image_frame.get(ann["image_id"])
        if frame is None:
            continue
        cls = categories.get(ann["category_id"])
        if cls is None:
            skipped += 1
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0 or h <= 0:
            issues.warning(f"degenerate box on frame {frame}: {ann['bbox']}")
            continue
        labels.frames[frame].append(Box(cls, x, y, w, h))
    if unknown_cats:
        issues.warning(f"skipped {skipped} boxes of unmapped categories {sorted(unknown_cats)} "
                       "(add aliases under box_classes in config/labels.yaml)")

    for frame, boxes in labels.frames.items():
        if sum(b.cls == "ball" for b in boxes) > 1:
            issues.warning(f"frame {frame} has more than one ball box")
    return labels, issues
