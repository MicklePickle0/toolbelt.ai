# Volleyball Stat Tracker (one-end-of-court, offline video analysis)

## Remote access: desktop from the laptop (VS Code Remote Tunnel; checked 2026-10-29)
The desktop (`DESKTOP-6LEKOOH`, user `michael`, has the footage and the GPU) is driven from the laptop through a **VS Code Remote Tunnel**: it goes out through Microsoft's relay, so no port forwarding, Tailscale or SSH server is needed. The OpenSSH server is NOT installed on the desktop, so VS Code Remote-SSH will not connect.

State on 2026-10-29:
- **Tunnel:** `desktop-6lekooh`, connected, but started from the VS Code window, not installed as a service. It stops when that window closes or the PC restarts, until the one-time step below is done.
- **Sleep:** sleep and hibernate are off on AC power (good).

One-time setup on the desktop:
1. Make the tunnel survive restarts. In a terminal on the desktop run `code tunnel service install`, and sign in with the same GitHub or Microsoft account the laptop will use. `code tunnel status` should then show `"service_installed": true`.
2. Keep the PC awake: Settings > System > Power, Sleep = Never (already set). Avoid surprise restarts: Settings > Windows Update > set active hours, or pause updates for the tournament week.
3. The tunnel service starts only after Windows logs in, so after a restart someone has to log in at the desktop (or set up automatic login).

From the laptop, each time:
1. **Connect:**
   - In VS Code, install the **Remote - Tunnels** extension (Microsoft), then press F1 and run "Remote-Tunnels: Connect to Tunnel...".
   - Sign in with the same account and pick `desktop-6lekooh`.
   - Or, without installing VS Code, open `https://vscode.dev/tunnel/desktop-6lekooh` in a browser.
2. **Open the repo:** File > Open Folder, then `C:\Users\Michael\Setter-Eyes`. The window's bottom-left corner shows "Tunnel: desktop-6lekooh".
3. **Claude Code:** install the Claude Code extension when VS Code offers to install it "in Tunnel: desktop-6lekooh" (it runs on the desktop and sees `data/`, the GPU and the event DB), and log in once. Running `claude` in the remote terminal also works.
4. **Web app:**
   - In the remote terminal, run `.venv\Scripts\activate`, then `python scripts/serve_app.py`.
   - VS Code forwards port 8000 by itself (Ports tab), so open `http://localhost:8000` on the laptop.
   - Leave it running in that terminal. Closing the laptop doesn't stop it, but stopping the tunnel does.
5. **Long jobs:** run them through the app's queue, or in a terminal that stays open. The work runs on the desktop, so a dropped connection doesn't kill a job the app started. Reconnect and check the Queue page.

**Tailscale for the web app (set up and checked 2026-11-02):**
- Both machines are on the `MicklePickle0` tailnet: the desktop `desktop-6lekooh` is `100.104.23.18`, and the laptop `3ds` is `100.110.36.100`.
- "Run unattended" is on, so Tailscale works before anyone logs in to Windows.
- On the desktop, start the app with `python scripts/serve_app.py --host 100.104.23.18` in its own window, not VS Code's. That address answers only over Tailscale, not on the home Wi-Fi.
- On the laptop or phone (Tailscale app, same account), open `http://100.104.23.18:8000`. That loads from the laptop. `http://desktop-6lekooh:8000` did not: the name needs MagicDNS on in the Tailscale admin console and "Use Tailscale DNS" on the laptop.
- The app does not restart by itself after a reboot. Log in at the desktop, then start it again (or reach the desktop over the tunnel and start it there).
- Still to do: `code tunnel service install`, and pause Windows Update for the tournament week.

If it won't connect:
- **Tunnel down:** the desktop restarted or the tunnel service isn't installed. Someone at the desktop runs `code tunnel status`, then `code tunnel service install`, or just opens VS Code.
- **Wrong account:** the laptop must sign in with the account the tunnel was registered with.
- **Port 8000 not reachable:** in the Ports tab, add port 8000 by hand.

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
- Two modes per video (owner, 2026-11-16): **game mode** (the full pipeline) and **practice mode** (pass scores and set location/arc only; no rallies or score).
- Processing stays after the set, never live (owner chose this on 2026-11-16, also for the cloud). Rotation is never a hard rule for identity: a missed serve or an unnoticed mis-rotation would shift every later rally.
- Goal: hands-free. Sanity checks (legal set scores, stat consistency) lower confidence so that only doubtful rallies need review.

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

Owner's rules (2026-11-01):
- Height counts: a pass in the 4 spot that is very low is a 1 (`low_pass` in `config/pass_scale.yaml`).
- An overpass on serve receive is automatically a 0.
- An overpass on a dig doesn't count toward the player's passing average.
- An overpass that our side then sets is a receive.

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

## Roadmap (revised 2026-11-16; details and measurements in docs/todo.md)
The Ottawa roadmap (Phases A-D, 2026-10-28 to 11-05) is in git history and `docs/todo.md`.
Goal now: hands-free. Most rallies and stats right without review, and as much data from
each video as it can give. Accuracy first, then the cloud (owner, 2026-11-16).

Ordering principle: what the camera measures well comes first. Side to side (along the net),
timing and height near the net are reliable from one low camera. Distance in depth is not,
so anything that needs it waits for Phase 2's depth work. (Owner) = only the owner can do it.

**Phase 1: trust the numbers (CPU, no new labels).**
1. Match format per game: best of 3 or 5, points per set as a list, win by 2, optional cap
   (presets and custom, like the net height).
2. Score and stat sanity checks: legal set ends, one serve a rally, serving team follows the
   point winners, kills <= attacks, at most 3 touches a side. A failed check lowers the
   confidence of the rallies involved, so only doubtful rallies need review.
3. Rally-step fixes (old A15): a serve needs a net crossing or enough confidence, plus
   the other three rules in `docs/todo.md`.
4. Re-run every processed set with the current pipeline (old A17), then measure how many of
   our touches have a located, named toucher (today 479 of 3551 touches in the DB).
5. Retrain the touch scorer on all reviewed rallies (old A18/C3), and test on a held-out gym.

**Phase 2: who and where (identity and 3D).**
1. Ball depth fixed at touches (the toucher's feet), bounces and the net.
2. Linking a player's track pieces within a rally without rotation: one number on one
   player at a time, sprint speed limit, number reads carried across an attack (old B5).
3. (Owner decides) Test SAM 2 / SAMURAI on `pakmen-black_set2`'s 24 labelled attacks
   (~1 h GPU), then ~3-6 min a set if it pays off.
4. Fix a touch's player in review; one fix relabels the whole track (old B7).
5. Self-labelling (old C1): confident ball arcs become detector labels, so new gyms
   don't need hand labels.

**Phase 3: setting and situational stats (mostly depth-free; can start alongside Phase 2).**
1. Set maps on Stats: where sets are played from, and where they go. The pass map stays.
2. Set type detection. Several cues each score every call, combined into one calibrated
   probability; cues that disagree send the set to review. The cues: along-net contact
   point, tempo, apex height, hitter, setter facing. (Owner) fills the table of calls:
   tempo, go, shoot, 34, 4, 52, 51, 50, 60, red, RS tempo, pipe, bic, shoulders.
3. Situational splits as SQL with counts:
   - setter in vs out of system (draft: pass 3-4; owner to confirm);
   - passer by receive zone;
   - hitter efficiency by zone, set type, system and setter.
4. Practice mode: pass scores and set location/type only, chosen per video at upload.
5. Distance-off-the-net features (set depth, front vs back row from depth) once Phase 2.1 lands.

**Phase 4: cloud (after Phases 1-2 are measured good enough).**
1. Upload a file or a link in the app (old A7).
2. Container, job queue, storage bucket, GPU workers on AWS, Azure or another provider.
   Target: ~10-20 min a set, after the set (never live). The home GPU stays as a fallback.
3. Login before anything is exposed beyond Tailscale (old D9).

**Phase 5: later.**
- Multi-frame ball detector (TrackNetV3 / WASB style).
- Actions with pose (dump, free ball vs attack, block; old C4), then a learned action
  classifier from corrections (old C5).
- Rotation and libero as a soft hint only (old B6); roster and lineup entry (old B1).
- Serve-receive by zone maps (milestone 11), serve placement (milestone 12), set arc
  overlay and side view (milestone 10).
- Combos and trends across games; DataVolley export.
- Third-set side switch (old A10); the player app; renaming the package to Toolbelt.

Still open from Ottawa:
- (Owner) `code tunnel service install`.
- (Owner) Calibrate `416-knight_set1`, `mac-platinum_set1` and `pakmen-black_set3`.
- (Owner) How scrappy non-setter second touches are counted.

## Product shape (later; do not build yet)
Decided 2026-10-27, for further down the road. Perfect the local pipeline first (accuracy,
identity, stats, exports, speed); no paid cloud hosting (AWS/Azure/GCP) until then. Until
then, keep the code easy to move later: an API in front of the event DB, video stored apart
from the code, and processing that runs as a batch job.
- Hudl/Balltime-style: upload video from any camera (no recording app first), process
  offline on a GPU worker, view in two front ends over one FastAPI backend and event DB.
- Coaches: web app for laptop and iPad. Review queue, roster/lineup, stats, exports
  (DataVolley `.dvw`, CSV, PDF stat sheet, MP4 reels).
- Players: phone app (a home-screen web app first, native later). Their own touches,
  highlights, stats, sharing.
- Clips are time ranges over the original video; cut an MP4 with ffmpeg only on
  download/share.
- Players are minors: invite-only teams, consent before clips are shared, coach approval
  of what players can share. Check the privacy laws that apply.
- Order: (1) local only, as now; (2) private beta: hosted web app, with the owner's desktop
  GPU pulling jobs from a queue (no cloud GPU bill); (3) cloud GPU workers, player app,
  then a capture app that checks camera placement (behind our end line, court in frame).
- Before paying for cloud GPUs: 1080p input (downscaled on the phone, so uploads are
  smaller too) and faster inference (see `docs/todo.md`).

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
- Document every issue run into and how it was fixed (owner, 2026-10-31): a row in
  `docs/issues-log.md` (problem, cause, fix, where) as soon as it is understood, including
  dead ends and causes not found yet (left under "Still open" until fixed).
- Ask before adding audio, live processing, or far-side analysis; those are out of scope.
- Protect `data/` and `models/` (2026-10-26: a junction inside a temporary git worktree plus
  `git worktree remove --force` deleted both; only a Windows snapshot saved the hand
  labels). Never link real
  data into a folder that will be deleted; to run old code on real data, point PYTHONPATH
  at the old checkout's `src` instead. Ask the owner before any recursive or forced delete
  outside the scratchpad, and list the target (and any links in it) first.
