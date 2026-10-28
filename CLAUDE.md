# Volleyball Stat Tracker (one-end-of-court, offline video analysis)

Product name: **Toolbelt** (owner, 2026-10-28; formerly Setter Eyes). Use it in anything a user sees (web app, docs, exports). The repo folder and the Python package `setter_eyes` keep the old name for now; renaming them is a separate change to ask about.

An automated alternative to Hudl / DataVolley / VolleyStation. Input: a match or training video filmed from low, behind one end of the court. Output: per-player stats for ONE team (the near side), a ball-path overlay, and an editable event database.

## Product constraints (decided, do not revisit without asking)
- Only the near side of the court is analyzed. The far side is too small/distant to be reliable.
- The pipeline does not decide the score alone. It tracks the score internally by inferring each point's winner from who serves next (`rally/winner.py`, stored as `point_winner_source = next_serve` with a confidence), and uses that winner to judge the HOW of the point: an attacker swung and their side scored -> kill, without having to see a fingertip block touch (owner, 2026-10-27). Scoreboard OCR (when visible) and the one-key "point us / point them" in the review UI override it. Vision infers rally start/end and the HOW of the point (kill, error, ace, etc.). Big blocks that win points matter; tiny block touches don't need to be found.
- NO audio features. Tournament audio is too distorted/loud.
- Processing is OFFLINE (whole rally available). Do not design for live labeling.
- Every automated decision carries a confidence score. Low-confidence items go to a review queue. Human corrections are stored and become training data.
- Pass quality (serve receive, first ball only) uses the 0-4 scale below. Thresholds live in config, never hardcoded.
- Camera angle varies per game (different diagonals, low to the ground). Do not assume a fixed viewpoint across games.

## Pipeline (build in this order, verify each visually before moving on)
1. **Ingest**: read video (OpenCV/ffmpeg), keep original fps + timestamps, cache frames/crops. Footage from 1080p30 up to 4K60: whole-video passes sample at 30 Hz (`video.sample_hz`) and pixel settings are written for 4K and scale with the video's width (`video.reference_width`).
2. **Court calibration**: automatic line fitting first (net tape, sidelines, end line, attack line). Output a confidence score. If confidence is low, or the user asks, fall back to a manual-click UI (6-8 known points). Solve camera pose with `cv2.solvePnP` (not just a floor homography). If the camera drifts, track from the initialization with feature tracking / optical flow. Expect better lateral than depth accuracy.
3. **Person detection + tracking**: RF-DETR (class: person), tracker with camera-motion compensation (BoT-SORT/ByteTrack via `supervision` as a starting point). Filter to near-side players by court position and jersey color. Floor position from the feet.
4. **Ball detection + tracking**: RF-DETR fine-tuned on our own footage at high input resolution (1280+) with class `ball`. Hand-written Kalman filter with gravity. Fit ballistic arcs per flight segment between contacts.
5. **3D ball estimate**: camera rays from calibration + gravity-constrained arc fit for metric scale. Ball apparent size and net height are cross-checks only, not the primary signal. Net height is a per-game setting (`games.net_height_m`, chosen at setup): presets 2.43 m (17-18U boys), 2.35 m (15-16U boys), 2.24 m (13-14U boys and women's), plus a custom value. Everything that uses tape height (calibration scale, block check, ball-height checks) reads this one setting; confirm it by drawing the net tape at that height over the video. Anchor with player feet on the floor. Needs only to be good enough for the 0-4 serve-receive scale. Optional overlay: vertical drop line from ball to its floor point plus a ground trail.
6. **Pose**: RTMPose (via `rtmlib`/ONNX) on player crops. Used for wrist/forearm proximity, platform arm cues, arm-swing velocity.
7. **Contact detection** (no audio): fuse soft evidence, never a single detector: trajectory break (direction/speed change), ball-to-wrist/forearm proximity, pose cues, rally-state priors (max 3 contacts per side plus a block). Each contact gets a confidence.
8. **Rally segmentation**: start from serve detection, end from ball landing / out / dead ball. Scoreboard OCR (macOS Vision, or EasyOCR/PaddleOCR on other OSes) is a sanity check only.
9. **Point outcome (the HOW)**, post-processed after the rally is complete from the final contact chain and the point winner:
   - our attack lands in / opponent cannot keep it in play -> kill
   - our attack in net, out, or blocked -> attack error
   - our reception out, into net, or unplayable -> reception error (shank)
   - our serve untouched on their side -> ace
   - our serve in net / out -> service error
   - ambiguous -> review queue, resolved against the point-winner ground truth
10. **Action classification** (after the rally resolves):
    - Stage 1, rules: touch order (serve -> receive -> set -> attack).
    - Stage 2, overrides using trajectory + pose: dump (second contact, usually setter, path crosses net), free ball/overpass (soft contact crossing net, no arm swing), attack (high arm-swing velocity, descending trajectory), block (contact at net, arms above tape, ball redirected), coverage (touch after a block, no resulting attack).
    - Stage 3: small learned classifier (gradient boosting or tiny temporal model) trained on corrected contacts from the review UI.
11. **Player identity + rotation** (see below).
12. **Review UI**: scrub timeline, fix a contact/action/player, set point winner, resolve queue. Corrections propagate across a tracklet.
13. **Stats + export**: SQL queries over the event DB; later a DataVolley-format export.

## Player identity design
Roster: user registers a team, then each player by jersey number + name. Each set, user enters the starting lineup (six numbers in rotation order).
1. Short-term tracking with camera-motion compensation.
2. Number voting: read numbers only on good crops (facing away, upright, large enough), accumulate votes per track. When a track passes a confidence threshold, bind it to a roster player and KEEP the binding when the number is not visible.
3. Appearance gallery: store embeddings (torso color, build) of confirmed players so a broken track can re-attach.
4. Lineup constraint: rotation state machine (six slots, rotate after each side-out). Each player appears at most once; positions follow rotation order. Libero swaps tracked separately (different jersey color helps).
5. Uncertain assignments go to the review queue; one correction relabels the whole tracklet.
Opponents are ignored for tracking.

## Serve-receive pass quality (0-4)
Measured on the FIRST ball after the serve only (not defense). Draft thresholds, to be tuned against labeled passes:

| Score | Meaning |
|---|---|
| 4 | Perfect: high and close to the net, in the setter target zone, setter barely moves; all four options (left, right/C, pipe, middle) plus an aggressive dump |
| 3 | Decent: about a meter off the attack line (owner's wording; confirm which side, likely between net and attack line), four options but the middle is harder to set |
| 2 | Mediocre: around or slightly behind the attack line; left, right (and pipe) only |
| 1 | Terrible: far off net and attack line; only out-of-system high ball to left or right |
| 0 | Ace |

Measure distance to a target point (depth AND lateral), not just distance to the net. Put thresholds in `config/pass_scale.yaml`. NOTE: the 3 vs 4 vs 2 boundaries above are an interpretation and must be confirmed against the owner's labeled examples.

## Data model (event DB: SQLite or DuckDB is the source of truth)
Tables (minimum): `games`, `sets`, `rallies`, `players` (roster), `tracks`, `contacts`, `ball_samples`, `corrections`.
- `contacts`: id, rally_id, t, frame, player_id (nullable), track_id, action, action_confidence, ball_x/y/z, outcome, pass_score (nullable), reviewed (bool).
- Never recompute stats from video; compute them with queries over `contacts`/`rallies` so combos (setter -> hitter kill %) and efficiency work without rerunning vision.

## Suggested repo layout
```
config/            pass_scale.yaml, court.yaml, model paths
src/ingest/  src/calibration/  src/detect/  src/track/  src/ball/
src/pose/  src/contacts/  src/rally/  src/identity/  src/classify/
src/db/  src/stats/  src/api/ (FastAPI)  src/ui/
data/ (videos/caches gitignored; labels tracked)  models/ (in-use models tracked, LFS)  tests/  scripts/
```

## Local setup
- Python 3.11 or 3.12 in an isolated env (`uv` or conda), git, ffmpeg.
- Core: `torch`/`torchvision` (CUDA build on NVIDIA; MPS on Apple Silicon), `opencv-python`, `numpy`, `scipy`.
- Vision: `rfdetr`, `supervision`, `rtmlib` + `onnxruntime` (`onnxruntime-gpu` on NVIDIA), `filterpy` optional (the ball Kalman filter should be hand-written).
- Number recognition / OCR: start with `easyocr`; fine-tune a small digit classifier later if needed.
- Data/stats: `duckdb` (or stdlib `sqlite3`), `pandas`.
- App: `fastapi`, `uvicorn`, plus Streamlit for a first review UI.
- Dev: `pytest`, `ruff`.
- Training (RF-DETR fine-tune at 1280): on the local GPU (RTX 4070 SUPER, 12 GB) via `scripts/train_detector.py`, or a rented GPU; never on a CPU.
- Verify current versions and licenses (especially RF-DETR model sizes) before committing.

## Data labeling and venue variety
Detector labels (`ball`, `person`): about 700-900 frames in total, NOT every frame. Neighboring frames are near-duplicates.
- Sample frames at random or evenly spaced times within each video (stratified so the whole set is covered), then drop near-duplicates (perceptual hash or embedding clustering).
- Add targeted frames: serve tosses, ball near the net, ball partly hidden, bright backgrounds, spare balls and wall-colored objects (hard negatives).
- Workflow: label ~200 frames, train a rough detector, use it to pre-label the rest, correct by hand, then prioritize frames the model is least sure about.
- Sample by venue, not by video, so one gym with many games does not dominate. Record the venue for every video in the `games` table.
- Contact/action labels are per touch (not per frame) and come from correcting pipeline proposals in the review UI.
- Videos are ~20 min per set. Run detection/tracking over the full video, store rally start/end times in the DB, and cut short clips with ffmpeg on demand. Do not physically split videos up front. Ball detection skips the time between rallies (play spans from a cheap motion score, `scripts/find_play.py`) and, while it follows a moving ball, detects only a window around it; both are checked against labelled rallies, never trusted blind.

Labeling tools (check current features and pricing before committing):
- Roboflow: dataset management, auto-labeling, click-assisted labeling (used by the original project).
- CVAT: free/open source, SAM-assisted labeling, interpolation between frames (label the ball every few frames and fill in).
- X-AnyLabeling: free desktop app, SAM click-to-segment, can pre-label with our own detector.
- Label Studio: open source, can connect our own model for pre-labels.
- FiftyOne: not a labeler; use it to curate, de-duplicate and find frames the model gets wrong.
- Click-to-label (SAM) suits players. The ball is ~20 px, so pre-label with the detector and fix by hand.

Background and venue variety:
- Detectors memorize venues. Variety in what is behind the ball and players (wall colors, lighting, camera height/angle, ball type, jersey colors, other balls and people in the background) matters more than raw frame count. Several venues with fewer frames each generalize better than one venue with many (roughly 4+ venues is a starting estimate, not a verified requirement).
- A new gym adds much more variety than the same gym on another day. Tournament footage is valuable (spare balls and other games behind the court are the hard cases).
- Hold out an entire venue, not only a game, to test generalization. Expect to label a small batch of frames (about 50-100, a rough guess) for a brand-new gym.
- With only 1-2 venues, use augmentation (brightness, color shift, blur, crop, scale), but it does not replace real venue variety.

## Milestones
1. Ingest + calibration (auto + manual fallback) + court overlay. Verify visually.
2. Player detection/tracking mapped to court coordinates.
3. Ball detection + Kalman + 3D arc fit + ball-path overlay (+ ground-line toggle).
4. Contact detection + rally segmentation + serve detection.
5. Event DB + point outcome logic + action classification (rules).
6. Review UI + corrections table.
7. Identity (roster, number voting, rotation).
8. Pass scoring (0-4) validated on hand-labeled passes.
9. Stats + DataVolley export + learned classifier from corrections.
10. Setting arc analysis: per set, release point, apex height, time to the hitter (tempo), landing/contact point relative to the antennas and net, and per-setter consistency by set type (left, right, middle, pipe, back set). Overlay the arc on the video and plot arcs on a side-view court diagram.
11. Serve-receive by zone: for each first ball after the serve, record the player, the court zone (1-6, near side) and floor point of the contact, and the 0-4 pass score. Per player: a court map of contact points colored by pass score, plus average score and error rate per zone, so it shows who passes best and worst in each position.
12. Serve placement: where our serves go on the opponent side, by zone (1-6) as percentages. Show a court map with each serve as a dot (colored by outcome: ace, in play, error) and a heatmap toggle. Landing point comes from the ball's 3D arc (floor crossing or the opponent's first contact). This is the ball only, not far-side player tracking, but far-side depth accuracy is weaker, so keep zones coarse, attach a confidence to each landing point and send low-confidence ones to the review queue.

## Roadmap (agreed 2026-10-28; progress and measurements live in docs/todo.md)
Deadline: 16U OVA Fall Classic, Ottawa, 2026-11-06/07. The owner has only a laptop and a phone there and wants each day's sets analysed the same day (6-9 sets on day 1). Flow: record each set at 1080p60, upload the file after the set (no live streaming), process it as a batch on the home PC (cloud GPUs as a backup), review and read stats in the web app over Tailscale. Live AI stats are out of scope. (Owner) = only the owner can do it.

**Phase A, Fall Classic minimum (by 2026-11-05):**
- A1 (Owner) Push after each session; the laptop only sees what is pushed.
- A2 (Owner) Calibrate the 10 Durham Attack sets by hand (`calibrate.py --manual`); auto calibration fails at those gyms.
- A3 (Owner) Tailscale on PC, laptop and phone; PC never sleeps; app starts on boot.
- A4 (Owner) Queue the Durham Attack sets overnight in the app.
- A5 Browser calibration page (tap 6-8 court points on a still).
- A6 Honest auto-calibration confidence (a wrong fit scored 0.87), so the app asks for taps.
- A7 Upload a video or paste a YouTube/Drive link in the app; it joins the manifest.
- A8 Speed-ups to ~30-40 min per 1080p set (~100 min measured 2026-10-28): people detection at 10 Hz, 2x2 tiles at 1080p (check accuracy, retrain if needed), TensorRT/fp16.
- A9 (Owner labels) ~50 ball frames from each of 2-3 Durham Attack gyms, then retrain.
- A10 Third-set side switch: detect the camera move, ask for a second calibration.
- A11 One-key point us/them and keyboard shortcuts on the review page.
- A12 Manual stat tracker on the phone, works offline (optional; owner picks the stats).
- A13 (Owner) Dry run ~2026-11-03/04 from the laptop away from home on Durham Attack sets.
- A14 (Owner) Recording plan: 1080p60, phone low behind our end line, spare battery, upload after each set.

**Phase B, per-player stats (after Ottawa, or earlier if Phase A goes well):**
- B1 (Owner) Roster (numbers + names) and each set's starting lineup.
- B2 Pick the toucher by hands at the ball (RTMPose), not nearest feet (wrong ~1/3 of labelled touches).
- B3 Jersey-number reading: PARSeq (needs the owner's OK to run torch-hub code) or our own digit model trained on ~300-500 crops the owner tags. Off-the-shelf EasyOCR read ~6/40.
- B4 Number votes per track; keep the binding when the number is hidden.
- B5 Re-attach broken tracks by appearance.
- B6 Rotation state machine and libero swaps.
- B7 Fix a touch's player in review; one fix relabels the whole track.
- B8 Pass scores in the pipeline (from the setter's feet).

**Phase C, accuracy and new gyms without labelling:**
- C1 Self-labelling: confident ball arcs become detector labels; retrain overnight on new footage.
- C2 Test on a whole held-out gym (Durham Attack); this defines "good enough without labels".
- C3 Retrain the touch scorer with reviewed rallies.
- C4 Action stage 2 with pose (dump, free ball vs attack, block).
- C5 Learned action classifier from corrections.
- C6 Rest of the dead-ball check (net shake); other courts' balls by size.
- C7 Serve-only rallies from an off-screen server; better inferred serves.
- C8 Stuff blocks checked on more footage.
- C9 Process the 3 remaining Humber sets.
- C10 Small known issues (mac-chrome_set2 camera jump, camera offsets, line judge marked as a player).

**Phase D, deeper stats and product:**
- D1 Clips per stat and highlight reels in the app.
- D2 Combos (setter-to-hitter kill %), trends across games.
- D3 (Owner decides) How scrappy non-setter second touches are counted.
- D4 Milestone 11, serve-receive by zone.
- D5 Milestone 12, serve placement.
- D6 Milestone 10, setting arcs.
- D7 DataVolley export.
- D8 Cloud GPUs (container, job queue, upload bucket; ~$5-10/day), so it doesn't need the home PC.
- D9 Login, if the app is ever exposed beyond Tailscale.
- D10 Optional: rename the package/repo to Toolbelt.
- D11 Practice-footage features (later).

**Order:**
1. 2026-10-28/29: A1-A4 (owner), A5-A7 (Claude).
2. 2026-10-30 to 11-01: A8 and A10 (Claude), A9 (owner), then retrain.
3. 2026-11-02: A11, then A12 if there is time. If still above ~40 min a set, start D8 as the backup.
4. 2026-11-03/04: A13 dry run, fix what breaks. 2026-11-05: freeze; no new features.
5. After Ottawa: Phase B top to bottom, with Phase C alongside (C1 first, so new gyms stop needing labels).
6. Then Phase D in the order that matters to the owner's coaching.

## Working across machines and conversations
Work happens on a desktop (footage, GPU, `data/`) and a laptop, in separate Claude Code
conversations that don't see each other. The shared state is the repo, so:
- Start by pulling, and read `docs/todo.md` (plan, measured results, unmerged branches),
  `docs/issues-log.md` (what broke and how it was fixed) and `docs/labelling.md`
  (labelling rules, including the owner's judgement calls).
- End by updating those docs with any decision or measured result, committing and pushing.
  Anything left only in a conversation or in Claude's local memory is lost to the other
  machine.
- Labels, calibrations, the manifest, the event DB and the models in use ARE in git (the
  detector through Git LFS: install `git lfs` on every machine). Videos, caches, frames,
  datasets and outputs are not: the laptop has no footage. Work that needs footage or the
  GPU is marked "on the desktop" in `docs/todo.md`. Commit labels as soon as they're done.

## Working rules for Claude Code
- Build one milestone at a time; add a visual verification script (overlay video/images) for each.
- Split train/val/test BY VIDEO, never by frame.
- Keep everything configurable and every automated output confidence-scored.
- Ask before adding audio, live processing, or far-side analysis; those are out of scope.
- Protect `data/` and `models/` (2026-10-26: a junction inside a temporary git worktree plus
  `git worktree remove --force` deleted both; only a Windows snapshot saved the hand
  labels). Never link real
  data into a folder that will be deleted; to run old code on real data, point PYTHONPATH
  at the old checkout's `src` instead. Ask the owner before any recursive or forced delete
  outside the scratchpad, and list the target (and any links in it) first.
