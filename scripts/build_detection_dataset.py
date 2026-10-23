"""Build the RF-DETR training dataset (Roboflow COCO layout) from box labels, split by video.

Usage:
  python scripts/build_detection_dataset.py [--out DIR] [--classes ball person] [--overwrite]

Then train on a rented GPU, e.g. RFDETRMedium().train(dataset_dir=OUT, resolution=1280, ...).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from setter_eyes import config
from setter_eyes.datasets import assign_splits, export_detection_dataset
from setter_eyes.labels import load_manifest, load_video_labels


def main() -> int:
    pipeline = config.load("pipeline")
    paths, ds = pipeline["paths"], pipeline["datasets"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--out", type=Path, default=config.repo_path(ds["detection_export_dir"]))
    ap.add_argument("--classes", nargs="+", default=ds["detection_classes"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    splits = assign_splits(manifest)
    videos = []
    for entry in manifest.videos:
        if entry.boxes is None:
            continue
        vl = load_video_labels(manifest, entry)
        if not vl.ok:
            print(f"skipping {entry.key}: label errors (run scripts/ingest_labels.py --check)")
            continue
        videos.append((entry, vl.boxes))
    if not videos:
        print("no videos with valid box labels")
        return 1

    summary = export_detection_dataset(
        args.out, videos, splits, args.classes, pipeline["ingest"]["jpeg_quality"],
        overwrite=args.overwrite, progress=print)
    for split, n in summary.images.items():
        counts = ", ".join(f"{c} {k}" for c, k in summary.boxes[split].items())
        print(f"{split:5}: {n} images ({counts})")
        if n == 0:
            print(f"  WARNING: {split} is empty; pin a video to it with 'split:' in the manifest")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
