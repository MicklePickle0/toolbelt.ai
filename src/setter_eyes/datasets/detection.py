"""Export labelled frames as an RF-DETR (Roboflow COCO layout) detection dataset.

    <out>/train/_annotations.coco.json + images
    <out>/valid/...
    <out>/test/...
    <out>/provenance.json   which video went to which split

Frames are decoded from the source video, so the dataset always matches the
labels' frame indices. Category ids are 1..N in config order.

With `tiles` (e.g. (3, 3)) each frame is cut into overlapping native-resolution tiles
(datasets/tiles.py) so the small ball isn't shrunk by the model's input resize. A tile
is kept if it fully contains a box; tiles that cut through a box are dropped (a half
ball with no label would teach the wrong thing); `negative_tiles` box-free tiles per
frame are kept as background examples.
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from setter_eyes.datasets.tiles import tile_boxes
from setter_eyes.ingest.video import Frame, frame_filename, read_frames, write_image
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
    tiles: tuple[int, int] | None = None,
    tile_overlap: float = 0.15,
    negative_tiles: int = 1,
    frame_cache: Path | None = None,
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
        rng = random.Random(entry.key)
        for frame in _frames(entry, sorted(labels.frames), frame_cache):
            h, w = frame.image.shape[:2]
            boxes = [bx for bx in labels.frames[frame.index] if bx.cls in cat_ids]
            # Boxes of non-training classes (other_ball): tiles touching them are left out,
            # so a stray ball is never shown to the model as background.
            avoid = [bx for bx in labels.frames[frame.index] if bx.cls not in cat_ids]
            for name, (x0, y0, tw, th), tile_boxes_ in _crops(entry.key, frame.index, w, h,
                                                              boxes, tiles, tile_overlap,
                                                              negative_tiles, rng, avoid):
                write_image(split_dir / name, frame.image[y0:y0 + th, x0:x0 + tw], jpeg_quality)
                image_id = len(b.images) + 1
                b.images.append({"id": image_id, "file_name": name, "width": tw, "height": th,
                                 "video": entry.key, "frame": frame.index,
                                 "tile": [x0, y0, tw, th]})
                for box in tile_boxes_:
                    b.annotations.append({
                        "id": len(b.annotations) + 1, "image_id": image_id,
                        "category_id": cat_ids[box.cls],
                        "bbox": [box.x - x0, box.y - y0, box.w, box.h], "area": box.w * box.h,
                        "iscrowd": 0,
                    })
                    boxes_count[split][box.cls] += 1

    for split, b in builders.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / ANNOTATIONS).write_text(json.dumps(b.to_coco()), encoding="utf-8")

    used = {entry.key: splits[entry.key] for entry, _ in videos}
    (out_dir / "provenance.json").write_text(
        json.dumps({"classes": classes, "videos": used,
                    "tiles": list(tiles) if tiles else None, "tile_overlap": tile_overlap},
                   indent=2), encoding="utf-8")
    return ExportSummary({s: len(b.images) for s, b in builders.items()}, boxes_count, used)


def _frames(entry: VideoEntry, indices: list[int], cache: Path | None):
    """Labelled frames, from data/cache/<id>/frames/ when cached (written with exact
    indices by extract_frames / sample_active_frames), else decoded from the video."""
    folder = cache / entry.key / "frames" if cache is not None else None
    missing = []
    for i in indices:
        path = folder / frame_filename(entry.key, i) if folder is not None else None
        image = cv2.imread(str(path)) if path is not None and path.exists() else None
        if image is None:
            missing.append(i)
        else:
            yield Frame(i, 0.0, image)
    if missing:
        yield from read_frames(entry.path, missing)


def _crops(key, index, w, h, boxes, tiles, overlap, negatives, rng, avoid=()):
    """-> [(file name, (x, y, w, h), boxes inside)]: the whole frame, or kept tiles."""
    if tiles is None:
        return [(frame_filename(key, index), (0, 0, w, h), boxes)]

    def touches(b, x0, y0, tw, th):
        return b.x < x0 + tw and b.x + b.w > x0 and b.y < y0 + th and b.y + b.h > y0

    out, empty = [], []
    for t, (x0, y0, tw, th) in enumerate(tile_boxes(w, h, tiles, overlap)):
        if any(touches(b, x0, y0, tw, th) for b in avoid):
            continue
        inside = [b for b in boxes if b.x >= x0 and b.y >= y0
                  and b.x + b.w <= x0 + tw and b.y + b.h <= y0 + th]
        touching = [b for b in boxes if touches(b, x0, y0, tw, th)]
        name = frame_filename(key, index).replace(".jpg", f"_t{t}.jpg")
        if inside and len(inside) == len(touching):
            out.append((name, (x0, y0, tw, th), inside))
        elif not touching:
            empty.append((name, (x0, y0, tw, th), []))
    return out + rng.sample(empty, min(negatives, len(empty)))
