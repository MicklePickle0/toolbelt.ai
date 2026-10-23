"""Pre-label cached frames with a trained detector, for correction in the labelling tool.

Runs on data/cache/<id>/frames/*.jpg and writes, per video:
  data/labels/<id>/prelabels.coco.json   COCO boxes (file names = the frame images); import
                                         into the labelling tool, correct, export as
                                         boxes.coco.json
  data/labels/<id>/prelabel_queue.csv    frames ordered least-certain first (label these next)

Frames already in that video's boxes.coco.json are skipped.

Usage:
  python scripts/prelabel.py --checkpoint models/rfdetr/<run>/checkpoint_best_total.pth \
      --video KEY [--video KEY ...] [--threshold 0.3]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from setter_eyes import config
from setter_eyes.labels import load_manifest
from setter_eyes.labels.boxes import frame_from_name

MODELS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
          "base": "RFDETRBase", "large": "RFDETRLarge"}


def labelled_frames(boxes: Path | None) -> set[int]:
    if boxes is None or not boxes.exists():
        return set()
    images = json.loads(boxes.read_text(encoding="utf-8"))["images"]
    return {f for img in images if (f := frame_from_name(img["file_name"])) is not None}


def main() -> int:
    pipeline = config.load("pipeline")
    tr, paths = pipeline["train"], pipeline["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--video", action="append", required=True, help="manifest key (repeatable)")
    ap.add_argument("--model", choices=sorted(MODELS), default=tr["model"])
    ap.add_argument("--resolution", type=int, default=tr["resolution"])
    ap.add_argument("--threshold", type=float, default=tr["prelabel_threshold"])
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    import rfdetr

    model = getattr(rfdetr, MODELS[args.model])(pretrain_weights=str(args.checkpoint),
                                               resolution=args.resolution)
    names = model.class_names
    cat_ids = {c: i for i, c in enumerate(pipeline["datasets"]["detection_classes"], start=1)}
    manifest = load_manifest(args.manifest)
    cache = config.repo_path(paths["cache_dir"])
    labels_root = args.manifest.parent / "labels"

    for key in args.video:
        entry = manifest.video(key)
        done = labelled_frames(entry.boxes)
        files = [f for f in sorted((cache / key / "frames").glob("*.jpg"))
                 if frame_from_name(f.name) not in done]
        images, anns, queue = [], [], []
        for i, path in enumerate(files, start=1):
            bgr = cv2.imread(str(path))
            h, w = bgr.shape[:2]
            det = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold=args.threshold,
                                include_source_image=False)
            images.append({"id": i, "file_name": path.name, "width": w, "height": h})
            ball_scores = [0.0]
            for (x1, y1, x2, y2), cls, score in zip(det.xyxy, det.class_id, det.confidence,
                                                    strict=True):
                name = names[int(cls)] if int(cls) < len(names) else str(cls)
                if name not in cat_ids:
                    continue
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cat_ids[name],
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0,
                             "score": round(float(score), 4)})
                if name == "ball":
                    ball_scores.append(float(score))
            best = max(ball_scores)
            # Most informative to label: no confident ball, or a borderline one.
            queue.append((round(1 - abs(2 * best - 1), 3), path.name, round(best, 3)))
            if i % 50 == 0:
                print(f"  {key}: {i}/{len(files)}")
        out = labels_root / key
        out.mkdir(parents=True, exist_ok=True)
        coco = {"images": images, "annotations": anns,
                "categories": [{"id": i, "name": c, "supercategory": "none"}
                               for c, i in cat_ids.items()]}
        (out / "prelabels.coco.json").write_text(json.dumps(coco), encoding="utf-8")
        with (out / "prelabel_queue.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uncertainty", "file_name", "best_ball_score"])
            w.writerows(sorted(queue, reverse=True))
        print(f"{key}: {len(images)} frames, {len(anns)} boxes -> {out / 'prelabels.coco.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
