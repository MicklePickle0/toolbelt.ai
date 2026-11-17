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
        det = cls(pretrain_weights=str(pretrained / name), resolution=resolution)
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


def ball_tiling(detect_cfg: dict, width: int) -> tuple[tuple[int, int], int]:
    """(tile grid, model input size) for a video this wide (`detect.ball_tiling`): the
    first entry whose min_width it reaches."""
    for t in detect_cfg["ball_tiling"]:
        if width >= t["min_width"]:
            cols, rows = (int(v) for v in str(t["grid"]).lower().split("x"))
            return (cols, rows), int(t["resolution"])
    raise ValueError(f"no detect.ball_tiling entry for width {width}")


def predict_tiled(model, bgr: np.ndarray, grid: tuple[int, int], overlap: float,
                  threshold: float, keep: dict[int, str]) -> list[tuple[str, list[float], float]]:
    """-> [(class, [x1, y1, x2, y2], score)] in full-image pixels."""
    h, w = bgr.shape[:2]
    return predict_windows(model, bgr, tile_boxes(w, h, grid, overlap), threshold, keep)


def predict_windows(model, bgr: np.ndarray, windows: list[tuple[int, int, int, int]],
                    threshold: float,
                    keep: dict[int, str]) -> list[tuple[str, list[float], float]]:
    """Detect in each window (x, y, w, h) of the frame in one GPU batch; overlapping
    windows' detections merged with NMS -> [(class, [x1, y1, x2, y2], score)] in
    full-image pixels."""
    origins = [(x, y) for x, y, _, _ in windows]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    crops = [np.ascontiguousarray(rgb[y:y + th, x:x + tw]) for x, y, tw, th in windows]
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


def predict_crop(model, bgr: np.ndarray, crop: tuple[int, int, int, int], threshold: float,
                 keep: dict[int, str]) -> list[tuple[str, list[float], float]]:
    """One tile-sized crop (x, y, w, h), e.g. around the ball
    -> [(class, [x1, y1, x2, y2], score)] in full-image pixels."""
    x0, y0, w, h = crop
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr[y0:y0 + h, x0:x0 + w]), cv2.COLOR_BGR2RGB)
    d = model.predict(rgb, threshold=threshold, include_source_image=False)
    return [(keep[int(c)], [float(x1 + x0), float(y1 + y0), float(x2 + x0), float(y2 + y0)],
             float(s)) for (x1, y1, x2, y2), c, s in zip(d.xyxy, d.class_id, d.confidence,
                                                         strict=True) if int(c) in keep]


def predict_full(model, bgr: np.ndarray, threshold: float,
                 keep: dict[int, str]) -> list[tuple[str, list[float], float]]:
    """Whole frame at the model's resolution (for people: large, and tiles would cut them)
    -> [(class, [x1, y1, x2, y2], score)] in full-image pixels."""
    d = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold=threshold,
                      include_source_image=False)
    return [(keep[int(c)], [float(v) for v in box], float(s))
            for box, c, s in zip(d.xyxy, d.class_id, d.confidence, strict=True)
            if int(c) in keep]
