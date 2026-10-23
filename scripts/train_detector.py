"""Fine-tune RF-DETR on the detection dataset, on this machine's GPU.

Build the dataset first (scripts/build_detection_dataset.py). Settings: `train:` in
config/pipeline.yaml; flags override them. Writes checkpoints and metrics to
models/rfdetr/<run>/. Point models.ball_detector in pipeline.yaml at the best checkpoint
(checkpoint_best_total.pth) when you're happy with it.

Usage:
  python scripts/train_detector.py [--model medium] [--epochs 60] [--resolution 1280] [--run NAME]
  python scripts/train_detector.py --epochs 1 --run smoke     # quick end-to-end check
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from setter_eyes import config

MODELS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
          "base": "RFDETRBase", "large": "RFDETRLarge"}


def main() -> int:
    # rfdetr prints rich tables; Windows' default cp1252 console/log encoding can't, and the
    # crash would end training after the first validation.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    pipeline = config.load("pipeline")
    tr = pipeline["train"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path,
                    default=config.repo_path(pipeline["datasets"]["detection_export_dir"]))
    ap.add_argument("--model", choices=sorted(MODELS), default=tr["model"])
    ap.add_argument("--resolution", type=int, default=tr["resolution"])
    ap.add_argument("--epochs", type=int, default=tr["epochs"])
    ap.add_argument("--batch-size", default=str(tr["batch_size"]))
    ap.add_argument("--grad-accum-steps", type=int, default=tr["grad_accum_steps"])
    ap.add_argument("--lr", type=float, default=tr["lr"])
    ap.add_argument("--run", help="output folder name (default: <date>_<model>_<resolution>)")
    args = ap.parse_args()

    for split in ("train", "valid"):
        ann = args.dataset / split / "_annotations.coco.json"
        if not ann.exists() or not json.loads(ann.read_text())["images"]:
            raise SystemExit(f"{ann} is missing or empty: run scripts/build_detection_dataset.py "
                             f"(and pin a labelled video to '{split}' in the manifest)")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU visible to torch; install the CUDA build (see README)")
    import rfdetr
    import rfdetr.config

    # Download COCO-pretrained weights into models/ (by default rfdetr uses the cwd).
    pretrained = config.repo_path(tr["pretrained_dir"])
    pretrained.mkdir(parents=True, exist_ok=True)
    name = getattr(rfdetr.config, f"{MODELS[args.model]}Config") \
        .model_fields["pretrain_weights"].default
    model = getattr(rfdetr, MODELS[args.model])(pretrain_weights=str(pretrained / name))

    run = args.run or f"{datetime.now():%Y-%m-%d_%H%M}_{args.model}_{args.resolution}"
    out = config.repo_path(tr["output_dir"]) / run
    batch = args.batch_size if args.batch_size == "auto" else int(args.batch_size)
    provenance = args.dataset / "provenance.json"
    print(f"training RF-DETR {args.model} at {args.resolution} on {torch.cuda.get_device_name(0)}"
          f" -> {out}")
    model.train(dataset_dir=str(args.dataset), output_dir=str(out), epochs=args.epochs,
                batch_size=batch, grad_accum_steps=args.grad_accum_steps, lr=args.lr,
                resolution=args.resolution, early_stopping=tr["early_stopping"],
                num_workers=tr["num_workers"], device="cuda",
                notes=json.loads(provenance.read_text()) if provenance.exists() else None)
    best = sorted(out.glob("checkpoint_best*.pth"))
    print("done. checkpoints:", *(str(p) for p in best or sorted(out.glob("*.pth"))), sep="\n  ")
    return 0


if __name__ == "__main__":       # required on Windows: data loader workers re-import this file
    raise SystemExit(main())
