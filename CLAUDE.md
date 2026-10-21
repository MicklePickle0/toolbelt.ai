# Volleyball Stat Tracker (one-end-of-court, offline video analysis)

An automated alternative to Hudl / DataVolley / VolleyStation. Input: a match or training video filmed from low, behind one end of the court. Output: per-player stats for ONE team (the near side), a ball-path overlay, and an editable event database.

## Product constraints (decided, do not revisit without asking)
- Only the near side of the court is analyzed. The far side is too small/distant to be reliable.
- The ML pipeline does NOT decide the score. Score comes from (a) scoreboard OCR when visible as a sanity check, and (b) a one-key "point us / point them" input in the review UI. Vision infers rally start/end and the HOW of the point (kill, error, ace, etc.).
- NO audio features. Tournament audio is too distorted/loud.
- Processing is OFFLINE (whole rally available). Do not design for live labeling.
- Every automated decision carries a confidence score. Low-confidence items go to a review queue. Human corrections are stored and become training data.
- Pass quality (serve receive, first ball only) uses the 0-4 scale below. Thresholds live in config, never hardcoded.
- Camera angle varies per game (different diagonals, low to the ground). Do not assume a fixed viewpoint across games.

## Pipeline (build in this order, verify each visually before moving on)
1. **Ingest**: read video (OpenCV/ffmpeg), keep original fps + timestamps, cache frames/crops.
2. **Court calibration**: automatic line fitting first (net tape, sidelines, end line, attack line). Output a confidence score. If confidence is low, or the user asks, fall back to a manual-click UI (6-8 known points). Solve camera pose with `cv2.solvePnP` (not just a floor homography). If the camera drifts, track from the initialization with feature tracking / optical flow. Expect better lateral than depth accuracy.
3. **Person detection + tracking**: RF-DETR (class: person), tracker with camera-motion compensation (BoT-SORT/ByteTrack via `supervision` as a starting point). Filter to near-side players by court position and jersey color. Floor position from the feet.
4. **Ball detection + tracking**: RF-DETR fine-tuned on our own footage at high input resolution (1280+) with class `ball`. Hand-written Kalman filter with gravity. Fit ballistic arcs per flight segment between contacts.
5. **3D ball estimate**: camera rays from calibration + gravity-constrained arc fit for metric scale. Ball apparent size and net height (2.43 m men's, 2.24 m women's) are cross-checks only, not the primary signal. Anchor with player feet on the floor. Needs only to be good enough for the 0-4 serve-receive scale. Optional overlay: vertical drop line from ball to its floor point plus a ground trail.
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
data/ (gitignored)  models/ (gitignored)  tests/  scripts/
```

## Local setup
- Python 3.11 or 3.12 in an isolated env (`uv` or conda), git, ffmpeg.
- Core: `torch`/`torchvision` (CUDA build on NVIDIA; MPS on Apple Silicon), `opencv-python`, `numpy`, `scipy`.
- Vision: `rfdetr`, `supervision`, `rtmlib` + `onnxruntime` (`onnxruntime-gpu` on NVIDIA), `filterpy` optional (the ball Kalman filter should be hand-written).
- Number recognition / OCR: start with `easyocr`; fine-tune a small digit classifier later if needed.
- Data/stats: `duckdb` (or stdlib `sqlite3`), `pandas`.
- App: `fastapi`, `uvicorn`, plus Streamlit for a first review UI.
- Dev: `pytest`, `ruff`.
- Training (RF-DETR fine-tune at 1280): use a rented GPU; do not train on a laptop CPU.
- Verify current versions and licenses (especially RF-DETR model sizes) before committing.

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

## Working rules for Claude Code
- Build one milestone at a time; add a visual verification script (overlay video/images) for each.
- Split train/val/test BY VIDEO, never by frame.
- Keep everything configurable and every automated output confidence-scored.
- Ask before adding audio, live processing, or far-side analysis; those are out of scope.
