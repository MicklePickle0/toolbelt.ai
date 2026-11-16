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

## Roadmap (revised 2026-11-16 after Ottawa; progress and measurements live in docs/todo.md)
The 16U OVA Fall Classic (Ottawa, 2026-11-06/07) has passed; the team placed well. The
Ottawa roadmap (Phases A-D) is in git history; its open items are carried below with their
old IDs. Goal now (owner, 2026-11-16): stats the owner can coach from, hands-free (most
rallies and stats right without review), and as much data from each video as it can give.
Accuracy and stats first, then the cloud. Processing stays after the set, never live.
Ordering principle: what one low camera measures well comes first (along the net, timing,
height near the net); anything that needs depth waits for F2. (Owner) = only the owner can
do it.

Standing: (Owner) push after each session (was A1). Recording plan stays as A14: 4K, phone
low and centred behind our end line, whole court in frame, process a 1080p60 copy.

**Phase E, stats you can trust now (CPU, no new labels).** Built on the laptop 2026-11-16
(E1-E5, E7-E9 done; E3 and E6 need the desktop's caches, below):
- E1 DONE (`match.py`, `data/match_formats.yaml`, Stats > Edit folders). Match format per game: best of 3 or 5 (25, 25, 15 / 25, 25, 25, 25, 15), points per
  set as a list, win by 2, optional cap; presets plus custom, like the net height.
- E2 DONE (`rally/sanity.py`, `scripts/check_sets.py`; review "Flagged" mode). Score and stat
  sanity checks (extends A15's impossible-score check): legal set ends,
  one serve a rally, serving team follows the point winners, kills <= attacks, at most 3
  touches a side, a receive only after their serve. A failed check lowers the confidence of
  the rallies involved so they rise in the review queue; nothing is fixed silently.
- E3 BUILT, NOT MEASURED (`rally.checks` in pipeline.yaml; needs the desktop's caches). Rally-step
  fixes (was A15): a serve needs a net crossing or enough confidence; a short
  piece before a sure serve is dropped; three touches by one side keep the rally.
- E4 DONE (keys listed on the review page). Review speed (was A11): review only what E2 flags; keys for point us/them, actions and
  team, next/previous touch, Enter = save & next. Reviewed stats are the right stats today.
- E5 DONE: a player fixed in review fills the track's other unconfirmed touches as a guess
  (0.9). Tracks reach the DB from the next re-run (E6). (was B7)
- E6 READY, RUN ON THE DESKTOP: `python scripts/rerun_sets.py` (backs up the DB, checks
  reviewed rallies survive, reports coverage). Re-run every processed set with the current pipeline (was A17, A4, C9): the Humber sets,
  `416-knight_set2/3`, `mac-platinum_set2`. Back up the DB first and check reviewed rallies
  survive. Then measure toucher coverage (today 479 of 3551 touches have a location).
- E7 DONE (Stats > Setting and situations). Set maps on Stats: where sets are played from (per setter) and where they go, next to
  the pass map.
- E8 DONE except set type (needs G1) (`stats/situational.py`). Situational splits, SQL with
  counts shown (was D2, D4):
  - setter in vs out of system (draft: pass 3-4; owner to confirm);
  - passer average and errors by receive zone;
  - hitter efficiency by zone, set type, system and setter;
  - trends across games.
- E9 DECIDED (owner, 2026-11-16): a non-setter's second touch counts as a set for whoever
  made it, with an assist when the attack kills (as the stat sheet already did). (was D3)

**Phase F, who and where (identity and 3D):**
- F1 Retrain the touch scorer on all reviewed rallies (was A18, C3); test on a held-out gym
  (was C2).
- F2 Ball depth fixed at touches (the toucher's feet), bounces and the net.
- F3 Track linking within a rally without rotation (was B5): one number on one player at a
  time, sprint speed limit, number reads carried across an attack. Attackers are the gap
  (3 of 24 named on `pakmen-black_set2`).
- F4 (Owner decides) SAM 2 / SAMURAI on `pakmen-black_set2`'s 24 labelled attacks (~1 h GPU);
  ~3-6 min a set if it pays off.
- F5 Self-labelling (was C1): confident ball arcs become detector labels; new gyms stop
  needing hand labels. (Owner) ~50 ball frames per new gym until then (was A9).
- F6 Actions with pose (was C4: dump, free ball vs attack, block), then a learned action
  classifier from corrections (was C5). Errors cascade: one missed touch shifts every later
  action in that side's run (clean rallies 94% actions right vs 67%).
- F7 Roster and starting lineup (was B1, owner); rotation and libero as a soft hint only,
  never a hard rule (was B6).

**Phase G, setting and practice (mostly depth-free; can run alongside F):**
- G1 Set type detection. Several cues each score every call, combined into one calibrated
  probability; cues that disagree send the set to review. The cues: along-net contact
  point, tempo, apex height, hitter, setter facing. (Owner) fills the table of calls:
  tempo, go, shoot (30/31/gap), 34, 4, 52, 51, 50, 60, red/flare, RS tempo, pipe, bic,
  shoulders.
- G2 Practice mode (was D11): pass scores and set location/type only, no rallies or score;
  chosen per video at upload.
- G3 Setting arcs (was D6, milestone 10): release, apex, tempo, landing vs antennas, per
  setter and set type; overlay and side view. Off-the-net distance after F2.
- G4 Serve-receive by zone maps (was D4, milestone 11) and serve placement (was D5,
  milestone 12, coarse zones with confidences).

**Phase H, cloud (after E and F measure good enough):**
- H1 Upload a file or a link in the app (was A7).
- H2 Container, job queue, storage bucket, GPU workers on AWS, Azure or another provider
  (was D8). Target ~10-20 min a set, after the set. The home GPU stays as a fallback.
- H3 Login before anything is exposed beyond Tailscale (was D9).

**Phase I, later:**
- I1 Multi-frame ball detector (TrackNetV3 / WASB style).
- I2 DataVolley export (was D7).
- I3 Third-set side switch (was A10).
- I4 Smaller accuracy items:
  - dead ball, net shake, other courts' balls (was C6);
  - off-screen servers (was C7);
  - stuff blocks (was C8);
  - known issues (was C10).
- I5 Manual stat tracker on the phone (was A12); player app; rename to Toolbelt (was D10).

Still open from Ottawa:
- (Owner) `code tunnel service install` (was A3).
- (Owner) Calibrate `416-knight_set1`, `mac-platinum_set1` and `pakmen-black_set3` (was A2;
  set 3 needs I3).

**Order (2026-11-16):**
1. Claude: E1-E3 (match format, sanity checks, rally fixes), then E6 re-runs (overnight
   GPU only for the sets missing pose).
2. Claude: E7-E8 set maps and situational splits on the reviewed sets, so the owner has
   coaching stats now; E4-E5 review speed alongside.
3. Owner: confirm "in system" (E8), fill the set-call table (G1), decide F4.
4. Claude: F1-F3, with G1-G2 alongside (set types and practice mode don't wait for depth).
5. Then H (cloud), then I in the order that matters to the owner's coaching.

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
