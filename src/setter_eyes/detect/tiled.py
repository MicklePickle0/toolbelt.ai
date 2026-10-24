"""RF-DETR detection on overlapping native-resolution tiles (datasets/tiles.py).

The ball is ~20-60 px in 4K footage; a whole frame fed to a 1280 px model shrinks it 3x,
so frames are cut into tiles, detected in one GPU batch, and merged with NMS.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from setter_eyes import config
from setter_eyes.datasets.tiles import tile_boxes

MODELS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
          "base": "RFDETRBase", "large": "RFDETRLarge"}


def load_detector(model: str, checkpoint: Path | None, resolution: int, fp16: bool = False):
    """A fine-tuned checkpoint, or the COCO-pretrained weights (downloaded into models/)."""
    import rfdetr
    import rfdetr.config

    cls = getattr(rfdetr, MODELS[model])
    if checkpoint is not None:
        det = cls(pretrain_weights=str(checkpoint), resolution=resolution)
    else:
        pretrained = config.repo_path(config.load("pipeline")["train"]["pretrained_dir"])
        pretrained.mkdir(parents=True, exist_ok=True)
        name = getattr(rfdetr.config, f"{MODELS[model]}Config") \
            .model_fields["pretrain_weights"].default
        det = cls(pretrain_weights=str(pretrained / name))
    if fp16:
        import torch

        # ~3x faster on the RTX 4070 SUPER with the same detections (checked on test frames).
        det.inference(compile=False, dtype=torch.float16)
    return det


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
