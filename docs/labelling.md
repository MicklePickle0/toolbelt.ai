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
python scripts/probe_video.py data/raw/match.mov
python scripts/normalize_video.py data/raw/match.mov data/videos/2026-10-01_vs_rivals.mp4

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

## Box labelling in batches (X-AnyLabeling or any image tool)

Frames come from `extract_frames.py --bursts` (runs of consecutive frames). Label a batch
spread over every video, then split the tool's single export back per video:

```bash
python scripts/label_batch.py make --name batch1 --per-video 2   # -> data/labels/batches/batch1/
# label in the tool, export COCO JSON, then:
python scripts/label_batch.py import data/labels/batches/batch1/<export>.json
python scripts/ingest_labels.py --check
```

- `import` merges into `data/labels/<id>/boxes.coco.json`; a frame labelled again replaces
  its old version. Each `make` skips frames already labelled or handed out in a batch.
- The manifest finds `labels/<id>/boxes.coco.json` and `labels/<id>/events.csv`
  automatically; `labels:` in an entry is only needed for other paths.
- Classes: `ball` (the match ball), `other_ball` (any ball not in play) and, later, `person`.
  Leave frames with no ball unboxed (negatives).
- In X-AnyLabeling: open the batch folder, draw rectangles, then export as COCO
  (rectangle) JSON. Keep the original file names.

## Manifest (`data/manifest.yaml`)

See [config/manifest.example.yaml](../config/manifest.example.yaml). Each video has:

- `id`: a stable key. Cache folders and DB rows hang off it.
- `path`: the video file.
- `venue` (required): the gym or tournament site. Names are compared ignoring case, spaces and punctuation, so `Queen's ARC` and `queens arc` are the same venue. Stored in `games.venue`.
- `net_height` (required): a preset from `net.height_presets` in [config/court.yaml](../config/court.yaml) (`boys_17_18u` 2.43 m, `boys_15_16u` 2.35 m, `boys_13_14u` 2.24 m, `women` 2.24 m) or a custom height in metres (e.g. `2.30`). Stored in `games.net_height_m`; calibration scale, block check and ball height all read it.
- `split` (optional): pins the video to `train`, `valid` or `test`. With venue splits this pins the whole venue; pinning two videos of one venue to different splits is an error.
- `labels.events` and `labels.boxes` (both optional; default `labels/<id>/events.csv` and `labels/<id>/boxes.coco.json` when those files exist).
- `lineups` (optional): six jerseys per set, in rotation order: the player starting in position 1 (who serves first when we serve first) first, then positions 2-6. Leave the libero out. With a lineup, each rally gets its rotation and our serves their server (`db/rotation.py`). For a labelled set, `python scripts/check_rotation.py --video <id>` suggests the lineup from who served.

Splits are never by frame. `datasets.split_by` in `config/pipeline.yaml` chooses the grouping:

- `venue` (default): every video from one venue goes to the same split, so valid/test score the model on gyms it has never seen.
- `video`: each video on its own. Use it while you have fewer than about three venues, otherwise one split gets everything. Switching between the two reshuffles the splits, so retrain afterwards.

Unpinned venues (or videos) are assigned from a hash of their name, using `datasets.split_ratios`. Adding videos or venues never moves an existing one to a different split.

The labels DB records its schema version. If `ingest_labels.py` says the DB is from an older schema, delete `data/labels.db` and run it again: it is rebuilt from the label files.

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

The quickest way to write `events.csv` is the keyboard labeller,
`python scripts/label_events.py --video KEY` (keys listed in the script's help).

### Judgement calls

- **Our serve starts off-screen:** label the hit, or the first frame the ball appears if the hit can't be seen. Put `rally_start` about 1 s before it.
- **A touch lasting several frames** (a set, a dig): one label, on the first frame of contact.
- **The ball hits the net:** no label; it only shows in how the rally ends (e.g. a serve into the net is a `service_error`).
- **Scramble touches:** label by touch order: 1st `dig`, 2nd `set` (even an ugly bump in mid-court), 3rd `attack` if swung, else `free_ball`. This is our own convention, not a DataVolley or Hudl rule.
- **Passing a free ball:** label it `dig`, after their `free_ball` or `overpass`. The stats count it as a free-ball pass, not a dig.
- **Our serve receive goes straight back over:** label it `overpass` (no pass score). It counts as a 0 pass in the passer's average.
- **A dig called for a ball-handling error:** label the `dig`, then end the rally with outcome `ball_handling_error`. It counts as an error, not a dig. If the dig was clean and the next touch was called, the dig still counts.
- **A replayed point** (a ball rolls onto the court, a referee decision): end the rally as
  usual at the whistle, but press `r` instead of a winner (`e` `r`). In the CSV:
  `outcome` = `replay`, `point_winner` blank. Nobody scores, the server doesn't change,
  and the rally is left out of all stats.
- **Timeouts:** no labels. Label rallies in an unbroken run, so the scorer can count fake rallies between them.

### Counting rule: second touches by non-setters (owner, 2026-11-16)

A scrappy second touch by someone who isn't the setter (the libero or an outside bump-sets or
hand-sets the second ball after a bad pass) is labelled `set` and counts as a set for whoever
made it, with an assist when the attack after it kills. Same as a setter's set.

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
- **Ball box**: tight around the visible ball, motion blur included. One `ball` per frame in match footage: the one in play, or between points the one being brought back for the next serve. More than one gives a warning; warm-up footage is fine.
- **Other balls** (spare balls, ball carts, other courts, warm-up balls): box them as `other_ball`. They are imported but not trained on, and tiles that touch them are left out of the training export, so a real-looking ball is never taught as background. If you can't tell which ball is the match ball, use the burst's neighbouring frames; if still unsure, `other_ball`.
- **Sampling**: label every Nth frame across many rallies and many videos, rather than dense runs from one rally. Neighbouring frames are near-duplicates.

## Where things go

| what | where |
|---|---|
| video registry | `data/manifest.yaml` |
| original recordings | `data/raw/` |
| normalised videos (label these) | `data/videos/` |
| label files | `data/labels/<id>/events.csv`, `data/labels/<id>/boxes.coco.json` |
| ground-truth events, ball centres | `data/labels.db` (same schema as the pipeline DB; rows `reviewed = 1`, confidence 1.0) |
| detection dataset | `data/datasets/detection/{train,valid,test}/_annotations.coco.json` + images, plus `provenance.json` |
| preview videos and stills | `data/outputs/` |
| sampled frames | `data/cache/<id>/frames/` |

Train on the local GPU with `python scripts/train_detector.py` (settings under `train:` in `config/pipeline.yaml`; RF-DETR 1.11 is Apache-2.0, resolution must be a multiple of 32, or 56 for `base`). Then pre-label the next frames with `python scripts/prelabel.py --checkpoint ... --video KEY`: it writes `prelabels.coco.json` to import into the labelling tool and `prelabel_queue.csv`, frames ordered least-certain first.
