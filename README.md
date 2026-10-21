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

Check the environment and run tests:

```bash
python scripts/check_env.py
pytest
ruff check .
```

## Layout

```
config/              court.yaml, pass_scale.yaml, pipeline.yaml (all thresholds live here)
src/setter_eyes/     ingest, calibration, detect, track, ball, pose, contacts,
                     rally, identity, classify, db, stats, api, ui
data/  models/       gitignored (videos, caches, event DB, weights)
tests/  scripts/
```

Train/val/test splits are made **by video**, never by frame. Model training (RF-DETR at 1280)
runs on a rented GPU.
