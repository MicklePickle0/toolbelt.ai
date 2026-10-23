"""Pre-label frames with a detector, for correction in the labelling tool.

Without --checkpoint the COCO-pretrained RF-DETR is used ("sports ball" -> ball, via
box_classes in config/labels.yaml): good enough to start the first batch. With a
fine-tuned checkpoint, its own classes are used.

Frames are cut into overlapping tiles (--tiles) so the ball (~20-60 px at 4K) isn't
shrunk to a few pixels by the model's input resize; detections are merged with NMS.

Batch mode (for X-AnyLabeling): writes <image>.json next to each image in the batch
folder, in X-AnyLabeling's own format, so boxes appear when the folder is opened.
Images that already have a .json (your work) are never overwritten. Also writes
prelabel_queue.csv there: frames ordered least-certain first.

Video mode: writes data/labels/<id>/prelabels.coco.json for the cached frames that
are not yet labelled, plus prelabel_queue.csv.

Usage:
  python scripts/prelabel.py --batch data/labels/batches/batch1
  python scripts/prelabel.py --checkpoint models/rfdetr/<run>/checkpoint_best_total.pth \
      --video KEY [--video KEY ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from setter_eyes import config
from setter_eyes.datasets.tiles import parse_grid, tile_boxes
from setter_eyes.labels import load_manifest
from setter_eyes.labels.boxes import frame_from_name

MODELS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
          "base": "RFDETRBase", "large": "RFDETRLarge"}


def load_model(args, tr: dict):
    import rfdetr
    import rfdetr.config

    cls = getattr(rfdetr, MODELS[args.model])
    if args.checkpoint:
        return cls(pretrain_weights=str(args.checkpoint), resolution=args.resolution)
    pretrained = config.repo_path(tr["pretrained_dir"])
    pretrained.mkdir(parents=True, exist_ok=True)
    name = getattr(rfdetr.config, f"{MODELS[args.model]}Config") \
        .model_fields["pretrain_weights"].default
    return cls(pretrain_weights=str(pretrained / name))


def model_ids(model, fine_tuned: bool) -> dict[int, str]:
    """Predicted class_id -> class name. Fine-tuned on our (Roboflow-layout) dataset, ids
    are 0-based positions in class_names; the COCO-pretrained model predicts COCO's own
    category ids (person = 1, sports ball = 37), not positions in its 80-name list."""
    if fine_tuned:
        return dict(enumerate(model.class_names))
    from rfdetr.assets.coco_classes import COCO_CLASSES

    return dict(COCO_CLASSES)


def canonical_names(ids: dict[int, str]) -> dict[int, str]:
    """class_id -> our class name (ball, person), via box_classes aliases."""
    aliases = {a.lower(): canon for canon, names in config.load("labels")["box_classes"].items()
               for a in [canon, *names]}
    return {i: aliases[n.lower()] for i, n in ids.items() if n.lower() in aliases}


def predict_tiled(model, bgr: np.ndarray, grid: tuple[int, int], overlap: float,
                  threshold: float, keep: dict[int, str]) -> list[tuple[str, list[float], float]]:
    """-> [(class, [x1, y1, x2, y2], score)] in full-image pixels."""
    h, w = bgr.shape[:2]
    tiles = tile_boxes(w, h, grid, overlap)
    origins = [(x, y) for x, y, _, _ in tiles]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    crops = [np.ascontiguousarray(rgb[y:y + th, x:x + tw]) for x, y, tw, th in tiles]
    dets = model.predict(crops, threshold=threshold, include_source_image=False)
    dets = dets if isinstance(dets, list) else [dets]
    found: dict[str, list[tuple[list[float], float]]] = {}
    for (x0, y0), d in zip(origins, dets, strict=True):
        for (x1, y1, x2, y2), c, s in zip(d.xyxy, d.class_id, d.confidence, strict=True):
            if int(c) in keep:
                found.setdefault(keep[int(c)], []).append(
                    ([float(x1 + x0), float(y1 + y0), float(x2 + x0), float(y2 + y0)], float(s)))
    out = []
    for cls, items in found.items():
        boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b, _ in items]
        scores = [s for _, s in items]
        for i in np.array(cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.5)).ravel():
            out.append((cls, items[int(i)][0], items[int(i)][1]))
    return out


def uncertainty(dets) -> tuple[float, float]:
    """(how useful to label next, best ball score): no confident ball or a borderline one."""
    best = max([s for c, _, s in dets if c == "ball"], default=0.0)
    return round(1 - abs(2 * best - 1), 3), round(best, 3)


def xanylabeling_json(name: str, w: int, h: int, dets) -> dict:
    shapes = [{"label": c, "shape_type": "rectangle", "flags": {}, "group_id": None,
               "description": f"prelabel {s:.2f}", "difficult": False, "attributes": {},
               "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}
              for c, (x1, y1, x2, y2), s in dets]
    return {"version": "4.2.0", "flags": {}, "shapes": shapes, "imagePath": name,
            "imageData": None, "imageHeight": h, "imageWidth": w}


def write_queue(path: Path, rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["uncertainty", "file_name", "best_ball_score"])
        wr.writerows(sorted(rows, reverse=True))


def main() -> int:
    pipeline = config.load("pipeline")
    tr, paths = pipeline["train"], pipeline["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="fine-tuned model (default: COCO-pretrained)")
    ap.add_argument("--batch", type=Path, help="batch folder (X-AnyLabeling mode)")
    ap.add_argument("--video", action="append", help="manifest key (repeatable)")
    ap.add_argument("--classes", nargs="+", default=["ball"], help="classes to pre-label")
    ap.add_argument("--model", choices=sorted(MODELS), default=tr["model"])
    ap.add_argument("--resolution", type=int, default=tr["resolution"])
    ap.add_argument("--threshold", type=float, default=tr["prelabel_threshold"])
    ds = pipeline["datasets"]
    ap.add_argument("--tiles", default=ds["tiles"] or "1x1", help="tile grid, columns x rows")
    ap.add_argument("--overlap", type=float, default=ds["tile_overlap"])
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()
    if not args.batch and not args.video:
        ap.error("give --batch DIR or --video KEY")

    model = load_model(args, tr)
    ids = model_ids(model, fine_tuned=args.checkpoint is not None)
    keep = {i: c for i, c in canonical_names(ids).items() if c in args.classes}
    if not keep:
        raise SystemExit(f"model classes {list(ids.values())[:5]}... map to none of {args.classes}")
    grid = parse_grid(args.tiles)

    def detect(path: Path):
        bgr = cv2.imread(str(path))
        return bgr.shape[1], bgr.shape[0], predict_tiled(model, bgr, grid, args.overlap,
                                                         args.threshold, keep)

    if args.batch:
        rows, written = [], 0
        images = sorted(args.batch.rglob("*.jpg"))
        for n, path in enumerate(images, start=1):
            out = path.with_suffix(".json")
            if out.exists():
                continue
            w, h, dets = detect(path)
            out.write_text(json.dumps(xanylabeling_json(path.name, w, h, dets)), encoding="utf-8")
            rows.append((*uncertainty(dets)[:1], path.name, uncertainty(dets)[1]))
            written += 1
            if n % 20 == 0:
                print(f"  {n}/{len(images)}", flush=True)
        write_queue(args.batch / "prelabel_queue.csv", rows)
        with_ball = sum(1 for r in rows if r[2] >= args.threshold)
        print(f"{written} frames pre-labelled in {args.batch} ({with_ball} with a ball box)")
        return 0

    manifest = load_manifest(args.manifest)
    cache = config.repo_path(paths["cache_dir"])
    cat_ids = {c: i for i, c in enumerate(pipeline["datasets"]["detection_classes"], start=1)}
    for key in args.video:
        entry = manifest.video(key)
        done = set()
        if entry.boxes is not None and entry.boxes.exists():
            done = {frame_from_name(i["file_name"])
                    for i in json.loads(entry.boxes.read_text(encoding="utf-8"))["images"]}
        files = [f for f in sorted((cache / key / "frames").glob("*.jpg"))
                 if frame_from_name(f.name) not in done]
        images, anns, rows = [], [], []
        for i, path in enumerate(files, start=1):
            w, h, dets = detect(path)
            images.append({"id": i, "file_name": path.name, "width": w, "height": h})
            for c, (x1, y1, x2, y2), s in dets:
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cat_ids[c],
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                             "iscrowd": 0, "score": round(s, 4)})
            u, best = uncertainty(dets)
            rows.append((u, path.name, best))
        out = args.manifest.parent / "labels" / key
        out.mkdir(parents=True, exist_ok=True)
        (out / "prelabels.coco.json").write_text(json.dumps(
            {"images": images, "annotations": anns,
             "categories": [{"id": i, "name": c, "supercategory": "none"}
                            for c, i in cat_ids.items()]}), encoding="utf-8")
        write_queue(out / "prelabel_queue.csv", rows)
        print(f"{key}: {len(images)} frames, {len(anns)} boxes -> {out / 'prelabels.coco.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
