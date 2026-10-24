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
different one. Inference runs on CPU; train on a GPU machine.

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

Train/val/test splits are made **by venue** (or by video while there are too few venues;
`datasets.split_by`), never by frame. RF-DETR training at 1280 runs on the local GPU
(`scripts/train_detector.py`) or a rented one.

Video decoding uses the GPU when available (`ingest.hw_decode`): the AV1 4K footage decodes
at ~20 fps on the CPU and 400+ fps on the RTX 4070 SUPER. Whole-video passes that don't need
full resolution use `iter_frames_scaled` (ffmpeg, GPU decode + scale).

## Court calibration and camera tracking (Milestone 1)

The Humber footage pans with play, so a video is calibrated once on one exact frame and
the pan is then tracked through every frame.

```bash
python scripts/calibrate.py --video mac-chrome_set1           # auto, else click ~8 points
python scripts/calibrate.py --video mac-chrome_set1 --manual  # straight to clicking
python scripts/track_camera.py --video mac-chrome_set1        # pose per frame + overlay preview
```

- Calibration: `data/calibration/<id>.json`; overlay and line mask in
  `data/outputs/calibration/`. Automatic fitting uses painted white lines and is honest
  about confidence; on courts bounded by floor-colour changes it falls back to clicking.
  Click UI: click the prompted point, `s` skip, `u` undo, `ijkl` nudge 1 px, Enter done.
- Tracking: `data/calibration/<id>_track.npz` (rotation + confidence per frame), checked
  by eye with `data/outputs/calibration/<id>_track.mp4` and `<id>_stills/`.
- Settings: `calibration:` in `config/court.yaml`.

## Ball tracking (Milestone 3)

```bash
python scripts/detect_ball.py --video titan-rise_set1        # every 2nd frame, ~1.6 h per 4K set
python scripts/track_ball.py --video titan-rise_set1         # tracks + 3D flight arcs, ~1 min
python scripts/render_overlay.py --video titan-rise_set1 --ball --ground --start 60 --seconds 60
```

- Detection: the tiled RF-DETR ball detector (`models.ball_detector`), fp16, resumable;
  `data/cache/<id>/ball_detections.npz`.
- Tracking: detections are re-expressed in the calibrated reference view (camera pan
  removed), joined into tracks with a constant-acceleration Kalman filter, and the
  match ball is the long, confident, moving track (spare balls never move). Short gaps
  are filled by smoothing; `data/cache/<id>/ball_path.npz`.
- Flights: split where the path suddenly changes course (touches, bounces), then fit in
  3D as `p0 + v0 t + g t^2 / 2` through the per-frame cameras; gravity fixes the depth.
  Each arc has a confidence; low ones go to review. `data/cache/<id>/ball_arcs.json`.
- Overlay: grey = detections, green = detected ball, orange = bridged, cyan = trail,
  white = fitted 3D flight, `--ground` = drop line and ground trail.
- Settings: `detect:` and `ball:` in `config/pipeline.yaml`. Needs a calibrated, tracked
  video (Milestone 1).

## Detector training (local GPU)

```bash
python scripts/build_detection_dataset.py --overwrite
python scripts/train_detector.py                      # settings: train: in pipeline.yaml
python scripts/prelabel.py --checkpoint models/rfdetr/<run>/checkpoint_best_total.pth \
    --video mac-chrome_set2                            # boxes to correct + least-sure-first queue
```
