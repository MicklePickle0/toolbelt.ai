# Labelled footage: formats and workflow

Labels are ground truth for two things:

- **Events** (`events.csv`): rallies, contacts, actions, jerseys, pass scores, point winners. Used to evaluate contact detection, rally segmentation, action classification and pass scoring, and later to train the learned classifier.
- **Boxes** (`boxes.coco.json`): ball and person bounding boxes on sampled frames. Used to fine-tune RF-DETR.

A video can have either kind of label, or both.

**The frame index is the shared key.** Frame `N` is the `N`-th decoded frame, counting from 0. Labels, cached frames, the event DB and pipeline output all use it.

## Workflow

```bash
# 0. Try the whole flow on synthetic data first (optional)
python scripts/make_demo_data.py

# 1. Check the recording. Phone footage is often variable frame rate (VFR) or has rotation
#    metadata. If it is flagged, normalise it and label the NORMALISED file.
python scripts/probe_video.py raw/match.mov
python scripts/normalize_video.py raw/match.mov data/videos/2026-10-01_vs_rivals.mp4

# 2. Register it: copy config/manifest.example.yaml to data/manifest.yaml and add an entry.

# 3. Label it (see the formats below). For box labelling in an image-based tool,
#    export frames with names the importer can map back to frame indices:
python scripts/extract_frames.py --video 2026-10-01_vs_rivals --every 15

# 4. Validate, then import into the labels DB (data/labels.db)
python scripts/ingest_labels.py --check
python scripts/ingest_labels.py

# 5. Check the labels by eye
python scripts/preview_labels.py --video 2026-10-01_vs_rivals --rally 1:4
python scripts/preview_labels.py --video 2026-10-01_vs_rivals --contacts   # one still per contact

# 6. Build the RF-DETR dataset (Roboflow COCO layout, split by video)
python scripts/build_detection_dataset.py --overwrite
```

`ingest_labels.py` reports every problem in every file. Errors block the import of that video; warnings don't. Running it again replaces that video's previous import.

## Manifest (`data/manifest.yaml`)

See [config/manifest.example.yaml](../config/manifest.example.yaml). Each video has:

- `id`: a stable key. Splits, cache folders and DB rows hang off it.
- `path`: the video file.
- `division`: sets the net height.
- `split` (optional): pins the video to `train`, `valid` or `test`.
- `labels.events` and `labels.boxes` (both optional).
- `lineups` (optional): six jerseys per set, in rotation order.

Videos with no `split` are assigned one from a hash of their `id`, using `datasets.split_ratios` in `config/pipeline.yaml`. Adding videos never moves an existing video to a different split. Splits are always by video, never by frame.

## Event labels (`events.csv`)

Write one row per event. Rows may be in any order: they are sorted by frame, and rows on the same frame keep their order in the file. Blank rows, and rows whose `type` starts with `#`, are skipped.

| column | used on | values |
|---|---|---|
| `set` | all | set number. Carried forward from the row above if blank (defaults to 1) |
| `frame` *or* `t` | all | frame index, or time as `12.5`, `1:02.25` or `1:01:02.25` (converted with the video's fps) |
| `type` | all | `rally_start`, `contact` or `rally_end` |
| `rally` | rally_start | optional rally number. Rallies are numbered 1, 2, ... per set by default |
| `team` | contact | `us` or `them` |
| `jersey` | contact (us) | jersey number. Leave it blank if you can't tell |
| `action` | contact | one of `actions` in [config/labels.yaml](../config/labels.yaml) |
| `pass_score` | contact | 0-4, **only** on our `receive` (the first ball after the opponent's serve) |
| `point_winner` | rally_end | `us` or `them` (required) |
| `outcome` | rally_end | one of `outcomes` in config/labels.yaml (optional) |
| `note` | any | free text |

Conventions:

- **rally_start**: the frame where the server starts the toss. **rally_end**: the frame where the ball lands, goes out, or the whistle makes it dead.
- **contact frame**: the first frame where the ball touches the player.
- Label opponent contacts too (`team: them`, no jersey). The point-outcome logic needs the whole contact chain.
- `receive` is serve reception only. Defensive first balls are `dig`. Pass scale: see CLAUDE.md and `config/pass_scale.yaml`.
- A block doesn't count toward a side's three contacts.

Example:

```csv
set,frame,t,type,team,jersey,action,pass_score,point_winner,outcome,note
1,1520,,rally_start,,,,,,,
1,1544,,contact,them,,serve,,,,
1,1590,,contact,us,7,receive,3,,,
1,1631,,contact,us,1,set,,,,
1,1668,,contact,us,9,attack,,,,
1,1702,,rally_end,,,,,us,kill,
```

These are **errors**:

- a contact or rally_end outside a rally
- a rally_start before the previous rally has ended
- an unknown action or outcome
- `pass_score` on anything other than our receive
- a frame outside the video
- a missing `point_winner`
- sets going backwards in time

These are **warnings**:

- a rally that doesn't start with a serve
- a `receive` that doesn't directly follow the opponent's serve
- more than three contacts by one side
- a jersey that isn't in the roster

## Box labels (`boxes.coco.json`)

Use standard COCO detection JSON, from any tool (CVAT, Roboflow, Label Studio).

- **Classes**: `ball` and `person`. Other names are mapped through the aliases under `box_classes` in config/labels.yaml (for example `Volleyball` becomes `ball`). Unmapped categories are skipped, with a warning.
- **Frame mapping**: an integer `frame`/`frame_id` field on the image, or a file name matching `frame_name_patterns`:
  - `<id>_f000123.jpg` (from `extract_frames.py`)
  - `frame_000123.png` (CVAT video tasks)
  - Roboflow's `_jpg.rf.<hash>` suffix is stripped first.
  - Don't use Roboflow's own video-to-frames upload: its frame names don't give a reliable index.
- **Resolution**: label at the video's original resolution. A size mismatch is an error. The ball is small, so don't downscale.
- **Negatives matter**: keep the frames you labelled where the ball is not visible. They are exported as images with no boxes.
- **Ball box**: tight around the visible ball, motion blur included. One ball per frame in match footage. More than one gives a warning; warm-up footage is fine.
- **Sampling**: label every Nth frame across many rallies and many videos, rather than dense runs from one rally. Neighbouring frames are near-duplicates.

## Where things go

| what | where |
|---|---|
| ground-truth events, ball centres | `data/labels.db` (same schema as the pipeline DB; rows `reviewed = 1`, confidence 1.0) |
| detection dataset | `data/datasets/detection/{train,valid,test}/_annotations.coco.json` + images, plus `provenance.json` |
| preview videos and stills | `data/outputs/` |
| sampled frames | `data/cache/<id>/frames/` |

Train on a rented GPU after `pip install -e ".[vision-gpu,train]"`. For example, `RFDETRMedium().train(dataset_dir="data/datasets/detection", resolution=1280, ...)`. Check the current RF-DETR model sizes and licences first.
