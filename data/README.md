# data/

Tracked in git: `manifest.yaml`, `videos.txt`, `setter_eyes.db` (holds review edits),
`labels/` (except frame images) and `calibration/`. Everything else (raw videos, normalised
videos, caches, frames, datasets, outputs, `labels.db`) is gitignored and rebuilt from the
videos and labels. Commit labels as soon as a labelling session is done (2026-10-26 they
existed only here and were nearly lost; `docs/issues-log.md`).
Workflow and label formats: [docs/labelling.md](../docs/labelling.md).

```
data/
  manifest.yaml            registry of videos, roster, label files (copy of config/manifest.example.yaml)
  raw/<gym>/               original recordings, one folder per venue (e.g. raw/humber/)
  videos/                  normalised, constant-fps videos; label and process THESE
  labels/batches/<name>/   frames gathered for one labelling session (scripts/label_batch.py)
  labels/<video-id>/
    events.csv             rallies, contacts, actions, pass scores, point winners
    boxes.coco.json        ball/person boxes exported from the labelling tool
  cache/<video-id>/frames/ sampled frames (scripts/extract_frames.py) for image-based labelling
  cache/<video-id>/        pipeline caches: activity.npz + play_spans.json (find_play.py),
                           ball/person detections, ball path and arcs, people tracks,
                           contacts and rallies
  outputs/                 preview videos and stills
  datasets/detection/      RF-DETR dataset built by scripts/build_detection_dataset.py
  labels.db                imported ground truth (scripts/ingest_labels.py)
  setter_eyes.db           pipeline event DB
```

Put each recording in the folder of the gym it was filmed in, and give it a manifest
`id` of `<opponent>_set<N>` (letters, digits, `_`, `-`), for example `mac-chrome_set1`.
Use that id as the frame-name prefix (`extract_frames.py --key`) and the folder name in
`labels/`. If you play the same opponent again on another day, prefix the date:
`2026-11-02_mac-chrome_set1`. Every manifest entry also needs `venue` (matching the gym
folder) and `net_height`.

Typical steps for a new recording:

```bash
python scripts/probe_video.py "data/raw/humber/MAC Chrome Set 1.webm"
# only if the probe flags VFR or rotation:
python scripts/normalize_video.py "data/raw/humber/MAC Chrome Set 1.webm" data/videos/mac-chrome_set1.mp4
# add an entry to data/manifest.yaml, then label into data/labels/mac-chrome_set1/
```
