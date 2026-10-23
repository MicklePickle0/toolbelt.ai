"""Label box batches across all videos in one folder, then split the export back per video.

make:   gather the next batch of unlabelled frames into data/labels/batches/<name>/<split>/
        (train/valid/test: the split of each frame's video, from the manifest):
        --per-video bursts (runs of consecutive frames) from every manifest video,
        spread over each video. With data/cache/<id>/bursts.json (sample_active_frames.py)
        only those bursts are used, --active-share of them from live play. Open that folder in the
        labelling tool (classes: ball, person) and export COCO JSON.
import: split a COCO export (or a folder of X-AnyLabeling .json files) by video
        (file names carry the video id) and merge it into
        data/labels/<id>/boxes.coco.json. Frames labelled again replace the old version.
        The manifest picks up labels/<id>/boxes.coco.json automatically.

Usage:
  python scripts/label_batch.py make --name batch1 [--per-video 2]
  python scripts/label_batch.py import data/labels/batches/batch1/annotations.json
  python scripts/label_batch.py import data/labels/batches/batch1      # reviewed folder
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from setter_eyes import config
from setter_eyes.datasets import assign_splits
from setter_eyes.labels import load_manifest
from setter_eyes.labels.boxes import frame_from_name

_SUFFIX = re.compile(r"_(?:jpe?g|png)\.rf\.[0-9a-fA-F]+$")


def bursts(indices: list[int]) -> list[list[int]]:
    """Group sorted frame indices into runs of consecutive frames."""
    runs: list[list[int]] = []
    for i in sorted(indices):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def spread(items: list, k: int) -> list:
    """k items evenly spaced through the list."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(step * j + step / 2)] for j in range(k)]


def taken(key: str, labels_root: Path) -> set[int]:
    """Frames already labelled or already handed out in an earlier batch."""
    out: set[int] = set()
    boxes = labels_root / key / "boxes.coco.json"
    if boxes.exists():
        out |= {f for img in json.loads(boxes.read_text(encoding="utf-8"))["images"]
                if (f := frame_from_name(img["file_name"])) is not None}
    for batch in (labels_root / "batches").glob("*"):
        out |= {frame_from_name(p.name) for p in batch.rglob(f"{key}_f*.jpg")}
    return out


def make(args, manifest, cache: Path, labels_root: Path) -> int:
    out = labels_root / "batches" / args.name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already has files; pick another --name")
    out.mkdir(parents=True, exist_ok=True)
    # Frames go to <batch>/<split>/, the split of their video (by match, never by frame).
    splits = assign_splits(manifest)
    total = 0
    owed_downtime = 0.0          # carried across videos so the batch-wide share holds
    for entry in manifest.videos:
        files = {frame_from_name(p.name): p
                 for p in (cache / entry.key / "frames").glob(f"{entry.key}_f*.jpg")}
        used = taken(entry.key, labels_root)
        listed = cache / entry.key / "bursts.json"
        if listed.exists():
            # Bursts picked by activity (sample_active_frames.py): keep the active share.
            spec = json.loads(listed.read_text(encoding="utf-8"))["bursts"]
            free = {"active": [], "downtime": []}
            for b in spec:
                run = list(range(b["start"], b["start"] + b["len"]))
                if all(f in files for f in run) and not set(run) & used:
                    free[b["kind"]].append(run)
            owed_downtime += args.per_video * (1 - args.active_share)
            n_down = min(len(free["downtime"]), int(owed_downtime + 1e-9))
            owed_downtime -= n_down
            n_active = min(len(free["active"]), args.per_video - n_down)
            chosen = spread(free["active"], n_active) + spread(free["downtime"], n_down)
            left = len(free["active"]) + len(free["downtime"]) - len(chosen)
        else:
            free_runs = [b for b in bursts(list(files)) if not set(b) & used]
            chosen = spread(free_runs, args.per_video)
            left = len(free_runs) - len(chosen)
        for run in chosen:
            for f in run:
                (out / splits[entry.key]).mkdir(exist_ok=True)
                src, dst = files[f], out / splits[entry.key] / files[f].name
                try:
                    os.link(src, dst)          # no extra disk space
                except OSError:
                    shutil.copy2(src, dst)
                total += 1
        print(f"  {entry.key} [{splits[entry.key]}]: {sum(len(r) for r in chosen)} frames "
              f"({left} bursts left)")
    print(f"{total} frames in {out} (train/valid/test subfolders)\n"
          "Review every frame (classes: ball, other_ball), then import the folder:\n"
          f"  python scripts/label_batch.py import {out}")
    return 0


def _video_of(file_name: str, keys: list[str]) -> str | None:
    stem = _SUFFIX.sub("", Path(file_name).stem)
    stem = re.sub(r"_f\d+$", "", stem)
    return stem if stem in keys else None


def _shape_box(shape: dict) -> tuple[float, float, float, float] | None:
    """Any X-AnyLabeling shape -> (x, y, w, h). Rectangles are the norm; circles (centre +
    edge point) and traced outlines (polygon, linestrip) become their bounding box."""
    pts = shape.get("points") or []
    kind = shape.get("shape_type")
    if kind == "circle" and len(pts) >= 2:
        (cx, cy), (ex, ey) = pts[0], pts[1]
        r = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
        return cx - r, cy - r, 2 * r, 2 * r
    if kind in ("rectangle", "polygon", "linestrip") and len(pts) >= 2:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    return None


def from_xanylabeling(folder: Path) -> dict:
    """X-AnyLabeling's per-image .json files (next to each frame) -> one COCO dict.
    A frame whose file has no shapes is a negative. Only rectangles are used."""
    images, anns, cat_ids = [], [], {}
    for path in sorted(folder.rglob("*.json")):
        if frame_from_name(path.stem + ".jpg") is None:
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": Path(d.get("imagePath") or
                                                         path.with_suffix(".jpg").name).name,
                       "width": d["imageWidth"], "height": d["imageHeight"]})
        for shape in d.get("shapes", []):
            box = _shape_box(shape)
            if box is None:
                continue
            x, y, w, h = box
            cid = cat_ids.setdefault(shape["label"], len(cat_ids) + 1)
            anns.append({"id": len(anns) + 1, "image_id": image_id, "category_id": cid,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    return {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n} for n, i in cat_ids.items()]}


def do_import(args, manifest, labels_root: Path) -> int:
    src = Path(args.export)
    data = from_xanylabeling(src) if src.is_dir() else json.loads(src.read_text(encoding="utf-8"))
    if src.is_dir():
        labels = {c["name"]: 0 for c in data["categories"]}
        for a in data["annotations"]:
            labels[data["categories"][a["category_id"] - 1]["name"]] += 1
        print(f"read {len(data['images'])} X-AnyLabeling files from {src}: {labels or 'no boxes'}")
    keys = [v.key for v in manifest.videos]
    cats = data.get("categories", [])
    by_image: dict[int, list[dict]] = {}
    for a in data.get("annotations", []):
        by_image.setdefault(a["image_id"], []).append(a)

    per_video: dict[str, list[dict]] = {}
    unknown = []
    for img in data["images"]:
        name = Path(img["file_name"]).name            # tools may prefix folders
        key = _video_of(name, keys)
        if key is None or frame_from_name(name) is None:
            unknown.append(name)
            continue
        per_video.setdefault(key, []).append({**img, "file_name": name})

    for key, images in per_video.items():
        path = labels_root / key / "boxes.coco.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
            {"images": [], "annotations": [], "categories": cats}
        new_names = {img["file_name"] for img in images}
        keep_ids = {i["id"] for i in old["images"] if i["file_name"] not in new_names}
        merged_images = [i for i in old["images"] if i["id"] in keep_ids]
        merged_anns = [a for a in old["annotations"] if a["image_id"] in keep_ids]
        # Re-number so ids stay unique; categories are matched by name on import.
        old_cats = {c["id"]: c["name"] for c in old.get("categories", [])}
        new_cats = {c["id"]: c["name"] for c in cats}
        names = sorted(set(old_cats.values()) | set(new_cats.values()))
        cat_id = {n: i for i, n in enumerate(names, start=1)}
        out_images, out_anns = [], []
        for src_images, src_anns, src_cats in ((merged_images, merged_anns, old_cats),
                                               (images, [a for img in images
                                                         for a in by_image.get(img["id"], [])],
                                                new_cats)):
            remap = {}
            for img in src_images:
                remap[img["id"]] = len(out_images) + 1
                out_images.append({**img, "id": remap[img["id"]]})
            for a in src_anns:
                out_anns.append({**a, "id": len(out_anns) + 1, "image_id": remap[a["image_id"]],
                                 "category_id": cat_id[src_cats[a["category_id"]]]})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"images": out_images, "annotations": out_anns,
                                    "categories": [{"id": i, "name": n, "supercategory": "none"}
                                                   for n, i in cat_id.items()]}),
                        encoding="utf-8")
        n_new = sum(len(by_image.get(img["id"], [])) for img in images)
        print(f"  {key}: +{len(images)} frames ({n_new} boxes) -> {path} "
              f"[{len(out_images)} frames total]")
    if unknown:
        print(f"WARNING: {len(unknown)} images not matched to a manifest video, e.g. {unknown[:3]}")
    print("next: python scripts/ingest_labels.py --check")
    return 0


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="gather the next batch of frames")
    mk.add_argument("--name", required=True)
    mk.add_argument("--per-video", type=int, default=2, help="bursts per video")
    mk.add_argument("--active-share", type=float,
                    default=config.load("pipeline")["activity"]["active_share"],
                    help="share of bursts from active play (needs sample_active_frames.py)")
    im = sub.add_parser("import", help="split a COCO export per video and merge it")
    im.add_argument("export", help="COCO export file, or a batch folder of X-AnyLabeling "
                    "per-image .json files (only once every frame in it is reviewed)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    labels_root = args.manifest.parent / "labels"
    if args.cmd == "make":
        return make(args, manifest, config.repo_path(paths["cache_dir"]), labels_root)
    return do_import(args, manifest, labels_root)


if __name__ == "__main__":
    raise SystemExit(main())
