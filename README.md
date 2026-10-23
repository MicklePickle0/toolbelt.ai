# Setter-Eyes

Offline volleyball stat tracker. Input: a match video filmed low from behind one end of the court.
Output: per-player stats for the near-side team, a ball-path overlay, and an editable event database.

Design, constraints and milestones: see [CLAUDE.md](CLAUDE.md).

## Setup

Requirements: Python 3.11 or 3.12, git, ffmpeg on `PATH`.

```bash
# with uv (recommended)
uv venv --python 3.12
uv pip install -e ".[dev]"            # core + tests
uv pip install -e ".[vision,ocr,app]" # full pipeline (CPU / Apple Silicon)

# NVIDIA: install the CUDA torch build first, then the gpu extra
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[vision-gpu,ocr,app,dev]"
```

No NVIDIA GPU (e.g. Intel Arc laptops): use the CPU torch build so `rfdetr` doesn't pull a
different one. Inference runs on CPU; training goes to a rented GPU anyway.

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[vision,ocr,app,dev]" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu
# deps pull extra OpenCV builds that clash on cv2; keep only contrib
uv pip uninstall opencv-python opencv-python-headless
uv pip install --reinstall-package opencv-contrib-python opencv-contrib-python
```

Check the environment and run tests:

```bash
python scripts/check_env.py
pytest
ruff check .
```

## Labelled footage

Register videos in `data/manifest.yaml`, label events (CSV) and boxes (COCO), then validate,
import, preview and export. Formats and the step-by-step workflow are in
[docs/labelling.md](docs/labelling.md). Smoke test on synthetic data:

```bash
python scripts/make_demo_data.py
python scripts/ingest_labels.py --manifest data/demo/manifest.yaml --db data/demo/labels.db
python scripts/preview_labels.py --manifest data/demo/manifest.yaml --video demo_a --contacts
python scripts/build_detection_dataset.py --manifest data/demo/manifest.yaml --out data/demo/detection
```

## Layout

```
config/              court.yaml, pass_scale.yaml, pipeline.yaml, labels.yaml (all thresholds and
                     vocabularies live here), manifest.example.yaml
src/setter_eyes/     ingest, calibration, detect, track, ball, pose, contacts,
                     rally, identity, classify, db, stats, api, ui,
                     labels (manifest, events CSV, COCO boxes, DB import), datasets (splits, export)
docs/                labelling.md
data/  models/       gitignored (videos, caches, event DB, weights)
tests/  scripts/
```

Train/val/test splits are made **by video**, never by frame. Model training (RF-DETR at 1280)
runs on a rented GPU.
