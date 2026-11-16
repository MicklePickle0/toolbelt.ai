# Toolbelt

Automated volleyball video analysis: an affordable alternative to Hudl, DataVolley and
VolleyStation. Film a match or practice from low behind one end of the court with a phone;
Toolbelt finds every rally and touch, names the player, grades serve-receive passes and
builds per-player stats, with a web app to review and correct what it found.

Only the near-side team is analysed. Processing is offline (after the set), runs on a local
NVIDIA GPU, and every automated decision carries a confidence score so that unsure items go
to a review queue. Corrections are stored and become training data.

Design, constraints and milestones: see [CLAUDE.md](CLAUDE.md). Plan and measured results:
[docs/todo.md](docs/todo.md). What broke and how it was fixed: [docs/issues-log.md](docs/issues-log.md).

(The repo folder and the Python package `setter_eyes` keep the project's old name.)

## How it works

| Step | What it does | Built with |
|---|---|---|
| Ingest | Reads 1080p30 to 4K60 footage, samples 30 times a second, GPU decoding | OpenCV, ffmpeg (NVDEC) |
| Play spans | Skips the time between rallies with a cheap motion score | NumPy |
| Court calibration | Camera pose from court points tapped in the app (automatic line fitting as a draft), then follows the camera as it pans | OpenCV `solvePnP`, optical flow |
| Ball detection | Fine-tuned detector on tiled frames; follows a moving ball with a single window | RF-DETR (PyTorch), fp16 |
| Ball tracking | Joins detections into tracks, drops spare balls, fits each flight as a 3D arc where gravity fixes the depth | Hand-written Kalman filter, SciPy least squares |
| People | Detects and tracks everyone, places them on the court by their feet, keeps our near-side players | RF-DETR (COCO `person`), `supervision` |
| Pose and jerseys | Keypoints for who touched the ball; back numbers read and voted per track, limited to the roster | RTMPose (`rtmlib`, ONNX Runtime GPU), PARSeq |
| Touches and rallies | Fuses trajectory breaks, hand-to-ball distance and rally rules into touches; finds serves, rally ends and the point winner from who serves next | Learned touch scorer (gradient boosting, scikit-learn) |
| Outcomes and actions | Serve, receive, set, attack, block, dig; kill, ace, error; 0-4 pass scores from where the setter takes the ball | Rules + `config/pass_scale.yaml` |
| Event database | Rallies and touches are the source of truth; stats are queries, never re-run vision | SQLite |
| Web app | Processing queue, calibration, rally review, stat sheet, pass map, search with clips and highlight reels | FastAPI, Uvicorn, vanilla JS |

Footage: 1080p at 30 fps is enough; anything up to 4K at 60 fps works too. The pipeline
samples 30 times a second either way, so higher settings mostly cost decoding time and disk
space, not extra detection passes. To run the whole pipeline over several videos (play
spans, ball and people detection, tracking, rallies), use `.\scripts\process_videos.ps1`,
or queue them in the web app.

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

## Web app

Videos and their processing queue, rally review and the stat sheet in a browser (phone
or laptop). Needs the `app` extra.

```bash
python scripts/serve_app.py                  # http://localhost:8000 on this PC
python scripts/serve_app.py --host 0.0.0.0   # reachable over Tailscale: http://<pc>:8000
python scripts/serve_app.py --no-worker      # review and stats only, no processing
```

Pages: Videos (queue and progress), Calibrate, Review (rally by rally: fix touches, actions,
players, pass scores and the point winner; Save & next), Stats (team and player
sheets, game summary, pass map), Search (player, action, result, pass score, set by; clips
and highlight reels with the player ringed). Sets can be grouped in tournament/game folders.

Each video is calibrated on its Calibrate page (tap the court points on a frame); processing
waits for it before following the camera. No login: only expose it on a private network
(Tailscale). Processing runs the
`app.steps` in `config/pipeline.yaml` one video at a time; logs in `data/outputs/logs/jobs/`.
The Streamlit review (`scripts/review.py`) still works and saves the same way.

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
data/  models/       labels, calibrations, event DB, models in use tracked (detector via Git LFS);
                     videos, caches, frames, datasets, outputs gitignored
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
python scripts/find_play.py --video titan-rise_set1          # play spans: skip time between rallies
python scripts/detect_ball.py --video titan-rise_set1        # 30 samples/s, play spans only
python scripts/track_ball.py --video titan-rise_set1         # tracks + 3D flight arcs, ~1 min
python scripts/render_overlay.py --video titan-rise_set1 --ball --ground --start 60 --seconds 60
```

- Footage: 1080p30 up to 4K60. Detection samples at 30 per second whatever the frame rate
  (`video.sample_hz`), and pixel settings are written for 4K and scale with the video's
  width (`video.reference_width`), so the same config serves both.
- Play spans: `find_play.py` scores on-court motion and writes
  `data/cache/<id>/play_spans.json`; detection skips everything outside it. With labelled
  rallies it lists any rally not fully covered: check that none are on a new venue.
  `detect_ball.py --all-frames` ignores the spans.
- Detection: the tiled RF-DETR ball detector (`models.ball_detector`), fp16, resumable;
  `data/cache/<id>/ball_detections.npz`. While a moving ball is followed, only one
  tile-sized window around it is detected instead of all 9 tiles; the whole frame is still
  scanned every 1/3 s, for new balls, and on any frame where the window misses
  (`detect.ball_follow`; `--no-follow` turns it off). The run ends by printing the model
  passes per frame.
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
