# Issues log

Problems hit while building Toolbelt (formerly Setter-Eyes), why they happened, and how they were fixed. Grouped
by pipeline stage, roughly in the order the stages were built. The last section lists what
is still open.

Figures come from the Humber footage (12 sets, 2026-10-17, 4K 59.94 fps). Touch and rally
scores come from the hand-labelled `titan-rise_set1` (42 rallies, 223 touches).

---

## 1. Video and decoding

| Problem | Cause | Fix | Where |
|---|---|---|---|
| Niagara RTW set 2 would not decode cleanly | Corrupt VP9 packets from about 6:18 on | Normalised once to `data/videos/niagara-rtw_set2.mp4`; the manifest points at that file, and it is labelled from that file | `scripts/normalize_video.py`, `data/manifest.yaml` |
| Whole-video passes were very slow | 4K AV1 decodes at about 20 fps on the CPU, and copying full 4K frames off the GPU costs about 40 ms each | GPU decode: OpenCV hardware acceleration for full frames, and ffmpeg `-hwaccel cuda` with `scale_cuda` (shrink on the GPU) for passes that don't need full resolution | `ingest/video.py` (`iter_frames_scaled`), `ingest.hw_decode` |
| Resumed detection could crash on its first save (Windows) | `np.load` keeps the previous `.npz` open, and Windows can't replace an open file | Read the previous run inside `with np.load(...)` | `detect/video.py` |
| Ball detection took ~1.6 h per 20-minute 4K set | Every 2nd frame, 9 tiles each, including all the time between rallies | Skip the time between rallies (play spans from a cheap motion score, generously padded, checked against labelled rallies); while a moving ball is followed, detect only one tile-sized window around it, with a full scan every 1/3 s and on any miss. Ball tracks end at skipped stretches | `scripts/find_play.py`, `rally/activity.py` (`play_spans`), `detect/follow.py`, `detect.ball_play_only`, `detect.ball_follow` |
| Settings assumed 4K at 59.94 fps | Pixel thresholds and frame strides were written for the Humber footage | Detection samples at `video.sample_hz` (30 Hz) whatever the frame rate; pixel settings scale with the video's width from `video.reference_width` (4K) | `config.py` (`scale_px`, `for_video`, `sample_stride`) |
| Frame numbers could differ between tools | Variable frame rate or rotation flags in some containers | Probe every video; files that need it are normalised, and the label check warns | `ingest/video.py` (`probe`), `labels/loader.py` |

## 2. Labelling (detector boxes)

| Problem | Cause | Fix | Where |
|---|---|---|---|
| Most extracted frames were downtime | Even sampling over a 20-minute set lands mostly between points | Sample by activity, in bursts of consecutive frames (some frames need motion context), stratified over the video | `scripts/extract_frames.py --bursts`, `scripts/sample_active_frames.py` |
| Spare balls and balls on other courts | Tournament gyms have many balls in view | A separate `other_ball` class, and dataset export avoids tiles that cut through them | `config/labels.yaml`, `datasets/detection.py` |
| Some batch-1 boxes were not rectangles | Polygon shapes from the labelling tool | Re-drawn as rectangles before import | (labelling fix) |
| Labelling on a different laptop | All data was on the training PC | Batches exported with train/valid/test folders, labelled in X-AnyLabeling on the laptop, COCO export brought back and checked | `scripts/label_batch.py` (make / import) |

## 3. Detector training (RF-DETR, local RTX 4070 SUPER)

| Problem | Cause | Fix | Where |
|---|---|---|---|
| Training crawled | Automatic batch size (9) overflowed 12 GB of GPU memory into system RAM | Batch 4 with 4 steps of gradient accumulation | `config/pipeline.yaml` `train` |
| Multi-scale training overflowed memory | Larger random input sizes at 1280 | Multi-scale turned off | `train.multi_scale` |
| `multi_scale: off` was ignored | YAML reads a bare `off` as false, not as the string the library expects | Quoted: `"off"` | `config/pipeline.yaml` |
| The PC crashed (bugcheck 0xEF, NVIDIA driver error) | GPU memory falling back to system memory under load | NVIDIA setting "Prefer No Sysmem Fallback" turned on; training resumed from the checkpoint | NVIDIA Control Panel |
| Training segfaulted after epoch 5 | Source files were edited while training ran | Rule: no code edits while training runs | working rule |
| Crashes printing to the Windows console | Non-ASCII characters in library output | Console reconfigured to UTF-8 | `scripts/train_detector.py` |
| 127-175 false boxes (heads, a basketball hoop) | Recolouring augmentation (meant to simulate other ball colours) taught it that round coloured things are balls | Recolouring turned off (`recolor_copies: 0`) | `config/pipeline.yaml` `datasets` |
| Pre-labels had the wrong classes | The COCO-pretrained model returns COCO category ids (person = 1, sports ball = 37); a fine-tuned model returns 0-based class indices | Class ids mapped per model | `detect/tiled.py` (`model_ids`) |

Result: ball detector at 96% recall on the test set, 90% on validation.

## 4. Court calibration (Milestone 1)

| Problem | Cause | Fix | Where |
|---|---|---|---|
| A median background over the video was useless | The camera pans with play | Calibrate on one exact frame, then follow the pan per frame | `calibration/track.py` |
| Automatic line fitting failed | Humber courts are bounded by floor-colour changes, not white lines | Manual clicks as the practical path (auto stays as a first try) | `scripts/calibrate.py` |
| Near end line placed about a foot too far forward | Its corners were covered, so the clicked corner was a guess | Click points anywhere along a line; the fit uses point-to-line distances, so corners aren't needed | `calibration/fit.py` (`line_click_distances`), `manual.py` |
| Calibrating 11 more videos one by one | — | `--missing --manual` clicks every uncalibrated video in turn | `scripts/calibrate.py` |

All 12 videos calibrated consistently (camera height 1.64-1.68 m).

## 5. Following the camera pan

| Problem | Cause | Fix | Where |
|---|---|---|---|
| Drift of degrees over a set | Chaining thousands of tiny frame-to-frame rotations random-walks | Track against keyframes; return to stored keyframes when the view comes back | `calibration/track.py` |
| Jumps to a wrong view | Repetitive texture (curtain panels, bleachers) matched one period off | Keyframe matches more than 1° from the flow estimate are rejected | `max_keyframe_jump_deg` |
| Summaries said "pan range 359°" | Subtracting rotation vectors breaks near 180°, where the vector flips sign | Relative rotation angle instead (real pans: 2-6°) | `scripts/track_camera.py` |
| Court overlay suddenly off by 40-180 px for the rest of a set (5 of 12 videos) | The camera was knocked or swung and the tracker re-locked with a constant error; re-solving showed the camera centre unchanged within about 1 cm | `check_camera.py` finds when (to 2 s) by comparing background features with the calibration frame. `fix_camera.py` carries the calibration across via floor features (SIFT + `solvePnP`), saves it as an extra calibration and re-anchors the track. No clicks needed | `calibration/alignment.py`, `transfer.py`, `track.py` (`reanchor`) |
| The first transfer gave a garbage pose | Passing the old pose as an initial guess to `solvePnPRansac` | Guess removed | `calibration/transfer.py` |

Fixed: `mac-chrome_set1`, `mac-chrome_set2` (first jump), `niagara-rtw_set2` (two jumps),
`reach-apex_set1`, `reach-apex_set2` (two jumps).

## 6. Ball tracking and 3D (Milestone 3)

| Problem | Cause | Fix | Where |
|---|---|---|---|
| Few touches had a 3D ball position | Flights between touches are short and often seen on one side only | Change-point thinning of flight splits; 3D from one confident side | `ball/arcs.py`, `contacts/events.py` |
| Serves were rarely seen being hit | Motion blur right in front of the camera, and a tiny ball 25 m away | Trace the first net-crossing flight back to just behind the end line; accept it at a hitting height and serve speed | `rally/segment.py` (`traced_serve`) |
| Trail drew straight jumps | Ball lost or tracker switched to another ball | Trail split at gaps and track switches | `ball/overlay.py` |
| pakmen-black_set2: 3D on 30% of touches (68-76% elsewhere), confident flights collapsing to 0-20% at 6-8 and 12-18 min, false "us attack" touches with image speeds of 57-139 m/s (2026-10-31) | A volleyball printed on a player poster on the gym wall, in view from that corner: 42% of the set's ball detections (score ~0.71). The "never moves" check judged a track's whole life: the poster's pan-free position wobbles 33-46 px over a minute with camera-track noise, over the 30 px limit at 1080p, and a track sitting on it for 40-100 s later picked up the real ball and outranked it (quality = samples x score). Detection itself was fine: the real ball was found as often there as elsewhere | Stillness judged per sample: still if within `static_spread_px` of its neighbours over +-`still_window_s` (1 s); poster wobble there is 7 px median, 21 px at the 95th percentile, real ball 37+ px at the 5th. Still samples add nothing to a track's quality and never enter the path. Re-tracked from the cached detections (no GPU): confident long flights 52-79% in every 2-min window; set 2 touches found 76 -> 87%, real 78 -> 88%, rallies 40 -> 45/45. Humber sets: no loss (titan actions 72 -> 77%, rallies 41 -> 42/42) | `ball/tracker.py` (`_moving`, `_quality`), `ball.still_window_s`; test `test_tracker_ignores_a_printed_ball_that_drifts` |

## 7. People tracking (Milestone 2)

| Problem | Cause | Fix | Where |
|---|---|---|---|
| Package conflict on install | `trackers` pulls `opencv-python`, clashing with `opencv-contrib-python` | Installed with `--no-deps` | `pyproject.toml` notes |
| Positions jumped when a player was hidden or misdetected | Missed detections and swapped boxes | Fill gaps from the samples before and after; replace outliers by interpolation | `track/people.py` (`clean_tracks`) |
| Joined tracks chained different people across the court | Linking broken tracks only by time and speed | Joins only within one area (our half, their half, off court) | `track/people.py` (`link_tracks`) |
| Spectators and coaches counted as players | Standing close to the court | Movement filter: people who never move are dropped | `track/roles.py` (`min_move_m`) |
| Green rings on spectators and opponents; libero and setter missing | Rings decided per whole track, but tracks swap teams at the net and the libero's tracks are short | Decided per moment from where the person is now (`ours_now`) | `track/roles.py` |
| Rings vanished when players jumped | Mid-jump the feet are off the floor, so the feet-to-floor position lands metres away | Jumps spotted by apparent height (box height × distance) inflating; those positions are interpolated | `track/people.py` (`apparent_height`) |
| Jump padding broke on short tracks | `np.convolve(..., "same")` misbehaves on arrays shorter than the kernel | `np.pad` + `"valid"` | `track/people.py` |

## 8. Touches and rallies (Milestone 4)

| Problem | Cause | Fix | Where |
|---|---|---|---|
| Touches found 57%, predictions real 51% | Nearly every touch had a candidate (220 of 223), but each touch has about two candidates, and course change + person distance didn't tell them apart (false ones scored higher) | A learned scorer trained on the labelled set: richer evidence per candidate (falling-then-rising, people close, height, turn angle, ball size), one touch kept per 0.35 s. Held-out rallies: **82% found, 88% real** | `contacts/features.py`, `contacts/scorer.py`, `scripts/train_touch_scorer.py` |
| Team wrong on 22% of touches | The ball's 3D side is unreliable (depth is the weak axis from behind one end line: 64% correct) | Learned team model; the ball's apparent size is the strongest cue. Held-out: **90%** | `contacts/scorer.py` |
| Touches lost when the ball was hidden at contact | The toucher often blocks the view at the moment of contact | Flights separated by up to 0.3 s still give a candidate | `contacts.max_bridge_s` |
| Two touches reported for one | A flight split into pieces | Merge within 0.12 s (heuristic); one per 0.35 s (scorer) | `contacts.merge_window_s`, `scorer.nms_s` |
| No way to measure touch quality | No ground truth | Hand-labelled `events.csv` plus a scorer: recall, precision, team, timing, misses by action, rally and serve errors | `contacts/evaluate.py`, `scripts/eval_contacts.py` |
| Fake rallies in timeouts went unnoticed | The scorer only judged predictions inside labelled rallies | Predicted rallies between labelled ones are now counted as false rallies | `contacts/evaluate.py` |
| Scoring a video the model trained on looked too good | In-sample | The report warns when the video was in the training set | `scripts/eval_contacts.py` |
| Team 78% and actions 50% on 1080p pakmen-black_set2 (2026-10-31) | The scorer's pixel features (ball size, image speeds) were raw pixels; it was trained on 4K, where the ball is twice as big (median 45.9 vs 24.5 px on the same video), so at 1080p every ball looked twice as far away (size is the team model's depth cue). The config scaling covered thresholds, not features | Pixel features are in 4K pixels: divided by `contacts.feature_unit_px`, which scales with the width. 4K unchanged (no retrain). Set 2: team 78 -> 85%, actions 50 -> 60%, winner 33 -> 39/44, outcome (inferred winner) 14 -> 21/44, but false rallies 2 -> 5. titan-rise_set1 at 1080p (mac-only scorer) now matches 4K on every line: found 83 -> 86% (4K 85%), outcome 21 -> 14/38 (4K 15/41) | `contacts/features.py`, `contacts.feature_unit_px` |

## 9. Labelling touches and rallies

| Problem | Cause | Fix | Where |
|---|---|---|---|
| VLC couldn't find the contact frame | No frame stepping backwards, no frame numbers | Keyboard labeller: frame-by-frame stepping, slow motion, frame number and time on screen, writes `events.csv` as you go | `scripts/label_events.py`, `labels/event_editor.py` |
| Seeking 4K AV1 frame by frame was too slow | Long gaps between keyframes in the original | A 1280-wide preview copy with a keyframe every 15 frames, checked to match the original frame for frame | `data/cache/<id>/proxy.mp4` |
| A set reviewed in the web app (pakmen-black_set2) couldn't be scored by `eval_contacts.py` | Reviews live in the event DB; the evaluation reads `events.csv` | `export_reviewed.py` writes the reviewed rallies (every touch confirmed) as `events.csv`; never overwrites. Rallies with no touches (a missed point added without them) are left out, so the pipeline's rally there counts as false | `scripts/export_reviewed.py` |
| Labels from review lean toward the pipeline: timing error 0.00 s | Proposed touches at the right time were kept as they were | Recall, precision and team are still fair; timing and rally start/end errors from reviewed sets aren't | - |

Labelling conventions decided along the way (also in `docs/labelling.md`, "Judgement calls"):

- **Our serve starting off-screen:** label the hit (or the first frame the ball appears); put the rally start about 1 s earlier.
- **Multi-frame touches (sets, digs):** one label, on the first frame of contact.
- **Ball hitting the net:** not labelled; it shows only in how the rally ends.
- **Scramble touches:** labelled by touch order (1st dig, 2nd set, 3rd attack or free ball). This is our convention; an earlier claim that DataVolley and Hudl define it this way could not be verified.
- **Free-ball passes:** labelled `dig` after their `free_ball` or `overpass`; the stats count them as free-ball passes, not digs.
- **Overpass on serve receive:** counts as a 0 pass and lowers the passer's average (`overpass_score` in `config/pass_scale.yaml`).
- **Uncontrolled dig:** a dig that ends the rally as a ball-handling error counts as an error, not a dig.
- **Timeouts:** not labelled; label rallies in an unbroken run so fake rallies can be counted.

## 9b. Rallies, review UI and later work (desktop, 2026-10-25/26)

| Problem | Cause | Fix | Where |
|---|---|---|---|
| Rallies 32/42 found, 8 false, serve found in 22 | Rallies came from ball visibility | Built from touches: touch groups < 3 s apart, serve before the first change of side, groups with no serve or exchange dropped | `rally/segment.py` (`segment_from_touches`) |
| Their unseen serve made our attack "the serve" | The serve hidden by a player or the tape | A rally opening with 2+ touches by one side was served by the other (inferred, lower confidence) | `rally/segment.py` |
| Touches marked ~5 frames early (an attack shown while the ball was still dropping) | A touch was placed at the last sample of the flight before it; a frame with the ball only bridged moved the flight split further off | Snap each candidate to the sharpest change of the ball's image velocity within 0.15 s. Timing 0.083 -> 0.017 s; touches found and real, actions and outcomes all up | `contacts/events.py` (`_sharpest_turn`), `contacts.snap_window_s` |
| Our serve placed at the toss (0.5-0.8 s early) and the hit counted as a "dig" | The ball is first seen big during the toss | The hit is where the ball stops rising (branch `our-serve-entry`) | `rally/serve_entry.py` |
| Free balls labelled as attacks | Rules counted touch order only | An "attack" launched upward > 4 m/s is a free ball | `classify/rules.py` |
| Replayed points had no label | `point_winner` was required | Outcome `replay`, no winner, left out of stats (labeller key `e` `r`) | `labels/events.py`, `labels/event_editor.py` |
| Review UI crashed on the second click | A cached SQLite connection used from another thread | A fresh connection per run | `ui/review_app.py` |
| After a merge, saving cut the rally back | Widgets kept the values typed before the merge | Widget keys include the rally's span and touch count | `ui/review_app.py` |
| Unchanged times logged as corrections | Times shown rounded | Untouched values keep their exact time | `ui/review_app.py` |
| Browser player showed whole seconds | HTML5 default controls | Own player: ms readout, frame stepping | `ui/player.py` |
| Re-running the pipeline would wipe reviews | The DB was rewritten per video | Reviewed rallies are kept; overlapping pipeline rallies dropped | `db/pipeline.py` |
| Play windows from motion saved little | Players move between points nearly as much as in play | Measured, not adopted (85-96% kept); see `docs/todo.md` | - |
| Pass position from the ball was wrong in depth | Depth is the weak axis from behind the end line | The setter's feet instead (branch `pass-scoring`) | `stats/pass_score.py` |

## 10. Tooling and process

| Problem | Fix |
|---|---|
| Shell heredocs mangled backslashes; a stray `cat` hung waiting for input | Multi-line edits written as scripts in a scratch folder and run from there |
| 2026-10-26: `data/` and `models/` deleted. Claude linked them (Windows junctions) into a temporary git worktree to compare old code, then `git worktree remove --force` followed the links. The SSD's TRIM made file-recovery tools useless (zeros) | Restored in full from a Windows Previous Versions snapshot (7:09 pm), copied out with robocopy (Explorer crashed on the many small files) and checked file by file. Labels, calibrations, the event DB and the models in use are now in git (detector via LFS); the rule against linking data into disposable folders is in CLAUDE.md |
| Small bugs: a shadowed variable (`plausible`), a dropped line (`finished += tracks`), `load_roles` needing a `Path` | Caught by tests or by checking output, and fixed |
| An unwanted change to `video.py` (log flushing) | Reverted at the owner's request |
| `data/setter_eyes.db` showed as changed after `--no-db` runs (2026-10-31) | Opening it (`db.connect`'s schema check) bumps SQLite's header change counter. Content compared row for row with the committed DB: identical; the committed bytes were restored |

---

## Still open

- **Rallies** (rebuilt from touches, out-of-fold): 39/42 found on titan-rise, 37/39 on
  mac-chrome; false rallies 0-9; inferred serves right ~half the time. Our own serve is
  the hard one (hit off-screen): 22/33 found; branch `our-serve-entry` gets 28/33.
- **Touches after the whistle:** the live-ball check (`rally/live.py`) catches 5 of 13
  with no real last touch cut; the ball-speed and net-shake signals are still to add
  (`docs/todo.md`).
- **Point outcome on pipeline touches:** ~half right (the rule itself is 39/42 on labelled
  touches); limited by the last touch's action, mostly blocks.
- **Blocks:** 7 of 18 found; a quick deflection at the net, often hidden by the tape.
- **The touch scorer** is trained on both labelled sets; cross-court it finds 70-83% of
  touches. The honest next test is a labelled set from a new venue.
- **Camera:** a second jump in the last ~25 s of `mac-chrome_set2` can't be fixed (no saved frame after it). `mac-iron_set1` and `reach-apex_set2` keep a small offset (20-40 px in 4K).
- **People:** a line judge on our sideline can get a ring; track ids break between points (player identity, Milestone 7, will re-link them).
- **Depth:** far-side and floor contacts have larger position errors.
- **Label check:** rally 1 of `titan-rise_set1` (0:19-0:25) has four touches in a row by them and their point by ball-handling error; probably a missing block.
- **pakmen-black_set2 after the poster and 1080p fixes:** false rallies went from 2 to 5 (2026-10-31; the rally step was re-run with the 1080p feature fix turned off to compare), plus 44/45 rallies found instead of 45/45. Actions 60%, team 85%.
  - **7:39 is real:** a missed point the owner added with no touches. The fix found it; the evaluation counts it false only because the label has no touches.
  - **8:56.8 and 14:12.3:** ball handling 3-8 s before the real serve (labelled 9:00.6 and 14:20.6). The fixed scorer now finds weak touches there (confidence 0.43-0.56) and calls one a serve. Because that piece has a "serve", the rule that drops a no-serve piece before a serve doesn't apply. It has 1-2 touches, lasts 1.4-2.9 s and has no net crossing. At 14:12 the team decoder then treats the next rally as continuing after that "serve", and flips the first touch of the real point.
  - **21:01 lost:** the teams are now right (our receive, set, attack, all `us`), but the serve isn't seen and the ball never crosses, so the rally step drops it. Before the fix, wrong teams made a fake crossing.
  - **Not new:** pre-serve "serve" pieces that split a real rally (15:36, 18:10, 20:27) existed before the fix. The evaluation doesn't count them because they overlap a labelled rally.
  - **Possible fixes, not built:** a serve needs a net crossing within ~2 s, or enough confidence; a short piece with no crossing just before a sure serve is dropped; a run of 3 touches by one side keeps its rally without a crossing. The poster should also become a hard negative for the ball detector (unboxed frames from this corner, A9).
