"""Export labelled frames as an RF-DETR (Roboflow COCO layout) detection dataset.

    <out>/train/_annotations.coco.json + images
    <out>/valid/...
    <out>/test/...
    <out>/provenance.json   which video went to which split

Frames are decoded from the source video, so the dataset always matches the
labels' frame indices. Category ids are 1..N in config order.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from setter_eyes.ingest.video import frame_filename, read_frames, write_image
from setter_eyes.labels.boxes import BoxLabels
from setter_eyes.labels.manifest import SPLITS, VideoEntry

ANNOTATIONS = "_annotations.coco.json"


@dataclass
class _SplitBuilder:
    categories: list[dict]
    images: list[dict] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)

    def to_coco(self) -> dict:
        return {"info": {"description": "setter-eyes detection export"},
                "categories": self.categories, "images": self.images,
                "annotations": self.annotations}


@dataclass(frozen=True)
class ExportSummary:
    images: dict[str, int]                 # split -> image count
    boxes: dict[str, dict[str, int]]       # split -> class -> box count
    videos: dict[str, str]                 # video key -> split


def export_detection_dataset(
    out_dir: str | Path,
    videos: list[tuple[VideoEntry, BoxLabels]],
    splits: dict[str, str],
    classes: list[str],
    jpeg_quality: int = 95,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ExportSummary:
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass overwrite=True to replace it")
        shutil.rmtree(out_dir)
    categories = [{"id": i, "name": c, "supercategory": "none"}
                  for i, c in enumerate(classes, start=1)]
    cat_ids = {c["name"]: c["id"] for c in categories}
    builders = {s: _SplitBuilder(categories) for s in SPLITS}
    boxes_count = {s: dict.fromkeys(classes, 0) for s in SPLITS}

    for entry, labels in videos:
        split = splits[entry.key]
        b = builders[split]
        split_dir = out_dir / split
        if progress:
            progress(f"{entry.key}: {len(labels.frames)} frames -> {split}")
        for frame in read_frames(entry.path, labels.frames):
            name = frame_filename(entry.key, frame.index)
            write_image(split_dir / name, frame.image, jpeg_quality)
            image_id = len(b.images) + 1
            h, w = frame.image.shape[:2]
            b.images.append({"id": image_id, "file_name": name, "width": w, "height": h,
                             "video": entry.key, "frame": frame.index})
            for box in labels.frames[frame.index]:
                if box.cls not in cat_ids:
                    continue
                b.annotations.append({
                    "id": len(b.annotations) + 1, "image_id": image_id,
                    "category_id": cat_ids[box.cls],
                    "bbox": [box.x, box.y, box.w, box.h], "area": box.w * box.h,
                    "iscrowd": 0,
                })
                boxes_count[split][box.cls] += 1

    for split, b in builders.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / ANNOTATIONS).write_text(json.dumps(b.to_coco()), encoding="utf-8")

    used = {entry.key: splits[entry.key] for entry, _ in videos}
    (out_dir / "provenance.json").write_text(
        json.dumps({"classes": classes, "videos": used}, indent=2), encoding="utf-8")
    return ExportSummary({s: len(b.images) for s, b in builders.items()}, boxes_count, used)
