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

## Roadmap (agreed 2026-10-28; progress and measurements live in docs/todo.md)
Deadline: 16U OVA Fall Classic, Ottawa, 2026-11-06/07. The owner has only a laptop and a phone there and wants each day's sets analysed the same day (6-9 sets on day 1). Flow: record each set in 4K (owner, 2026-11-01; keep the 4K files, since numbers read better), upload a 1080p60 copy after the set (no live streaming), process it as a batch on the home PC (cloud GPUs as a backup), review and read stats in the web app over the VS Code tunnel. Live AI stats are out of scope. (Owner) = only the owner can do it.

**Phase A, Fall Classic minimum (by 2026-11-05).** Status as of 2026-11-01:
- A1 (Owner) Push after each session; the laptop only sees what is pushed. Ongoing.
- A2 (Owner) Calibrate the Durham Attack sets by hand in the app. 7 done; still to do: `416-knight_set1`, `mac-platinum_set1`, `pakmen-black_set3` (the last needs A10).
- A3 DONE: VS Code Remote Tunnel for editing and Claude, plus Tailscale for the web app since 2026-11-02 (top of this file). Still to do: `code tunnel service install`.
- A4 (Owner) Queue the Durham Attack sets in the app. Processed: `pakmen-black_set1/2` only. Calibrated but not processed: `416-knight_set2/3`, `mac-platinum_set2`, `pakmen-gold_set1/2`.
- A5 DONE: browser calibration page.
- A6 DONE for now: automatic fits never count as calibrated (`app.accept_auto_calibration: false`).
- A7 Upload a video or paste a YouTube/Drive link in the app; it joins the manifest. Until then: drag the file into the VS Code explorer over the tunnel. Measure the upload size and time on the dry run: a 20-min 1080p60 file is several GB.
- A8 DONE: ~33 min per 20-min 1080p set (was ~2 h).
- A9 (Owner labels) ~50 ball frames from each of 2-3 Durham Attack gyms, then retrain at 1080p. Add the poster stretches of `pakmen-black_set2` as hard negatives, plus self-labels from its reviewed rallies (C1).
- A10 Third-set side switch: DEFERRED by the owner (2026-11-01). Most OVA tournaments have no side switch. Likely fix: find the switch, then process the two halves with their own calibrations and combine them.
- A11 Review speed (6-9 sets to review in a day): one key for point us/them, keys for actions and team, next/previous touch, Enter = save & next.
- A12 Manual stat tracker on the phone. Dropped for Ottawa unless time is left over.
- A13 (Owner) Dry run 2026-11-03/04 from the laptop away from home on Durham Attack sets.
- A14 (Owner) Recording plan: 4K (process a 1080p60 copy), phone low and CENTRED behind our end line (corner views cost accuracy at Durham Attack: worse depth, false entry serves), whole court in frame, no posters or windows behind the far court if there's a choice, spare battery, upload after each set.
- A15 (added 2026-10-31) Rally-step fixes from `pakmen-black_set2` (issues-log "Still open"):
  - a "serve" needs a net crossing soon after it, or enough confidence;
  - a short piece with no crossing just before a sure serve is dropped;
  - three touches by one side keep the rally without a crossing;
  - an impossible-score check (`pakmen-black_set1` inferred 33-17).
  CPU only; measure on set 2 and the two Humber sets. Not started (2026-11-01).
- A16 DONE 2026-10-31. Pass score and jersey number on each touch in review (none is editable in the web app today), so Ottawa gets pass averages and per-player stats from review even before the pipeline is right. Pass scores entered there become B8's labelled passes.
- A17 (added 2026-10-31) Re-run the processed sets with the 2026-10-31 fixes (poster, 1080p features, team decoder). CPU, from cached detections. First check that writing the event DB keeps every reviewed rally (`pakmen-black_set1/2`), and back up the DB. PARTLY DONE 2026-11-01: `pakmen-black_set1/2` re-run with pose_people (jerseys, pass scores), reviewed rallies kept, backups `data/setter_eyes.db.backup_2026-10-31_before_jerseys` and `..._2026-11-01_before_pass_spots`. The Humber sets still need pose_people + find_rallies.
- A18 (added 2026-10-31) Retrain the touch scorer with `pakmen-black_set2`'s reviewed rallies (C3, first Durham Attack data), then test on `pakmen-black_set1`'s reviewed rallies.

**Phase B, per-player stats (after Ottawa, or earlier if Phase A goes well):**
- B1 (Owner) Roster (numbers + names) and each set's starting lineup.
- B2 DONE 2026-10-31 (moved before Ottawa): toucher by forearms/hands at the ball (identity/toucher.py, RTMPose in scripts/pose_people.py).
- B3 DONE 2026-10-31: PARSeq (owner's OK) limited to the roster, on back views (identity/numbers.py). Fine-tune on reviewed crops later if needed.
- B4 DONE 2026-10-31, differently: tracks pass between players, so a touch takes its track's reads within +-4 s, and one number is on one player at a time. titan-rise_set1 at 1080p: 50% of our touches named, 91% right.
- B5 Re-attach broken tracks by appearance, holding a player's label as long as possible, even between clips (owner, 2026-11-01). Tried 2026-11-01, not built in: DINOv2 kNN 52%, per-video classifier 52-60%, elimination by on-court numbers (3-5 candidates left), speed-gated links (no change). Attackers are the gap: 3 of 24 named on `pakmen-black_set2`. Next try: an athlete re-ID model, 4K, or the starting lineup.
- B6 Rotation state machine and libero swaps. Also a server fallback from the rotation: if we serve and win the point, the same player serves again (owner, 2026-11-01). Done so far: the server is the deepest player who walks into view after the serve, and the libero never serves.
- B7 Fix a touch's player in review; one fix relabels the whole track.
- B8 DONE 2026-10-31: pass scores in the pipeline (from the setter's feet), with a confidence. On the owner's `pakmen-black_set2` labels: 69% exact, 92% within 1. Jerseys there: 47% of our touches named, 92% right.

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
- D1 PARTLY DONE 2026-11-01: Search tab (player, play, result, pass score, set by) with clips and highlight reels with the player ringed; game summary (set distribution, pass map) on Stats; tournament/game folders.
- D2 Combos (setter-to-hitter kill %), trends across games. (Search's set-by filter and the summary's sets-by-hitter cover the basics, 2026-11-01.)
- D3 (Owner decides) How scrappy non-setter second touches are counted.
- D4 Milestone 11, serve-receive by zone.
- D5 Milestone 12, serve placement.
- D6 Milestone 10, setting arcs.
- D7 DataVolley export.
- D8 Cloud GPUs (container, job queue, upload bucket; ~$5-10/day), so it doesn't need the home PC.
- D9 Login, if the app is ever exposed beyond Tailscale.
- D10 Optional: rename the package/repo to Toolbelt.
- D11 Practice-footage features (later).

**Order (revised 2026-11-01; the 2026-10-28 order slipped by ~2 days):**
1. 2026-11-01, done: A16, B8, B2-B4 (jerseys), D1 (folders, Search, reels, game summary), A17 for `pakmen-black_set1/2`.
2. 2026-11-01, owner:
   - restart the app (event DB schema v8);
   - A2 calibrate `416-knight_set1` and `mac-platinum_set1`;
   - A4 queue the 5 calibrated Durham Attack sets overnight (dry-run material; `pakmen-gold` is the new-gym test). Close other GPU programs first;
   - review `pakmen-black_set1`'s remaining 39 rallies (14/53 done) and dismiss the 4 false rallies on set 2;
   - put the sets in tournament/game folders (Stats > Edit folders);
   - `code tunnel service install`.
3. 2026-11-02, Claude:
   - A15 rally-step fixes;
   - A11 review shortcuts;
   - A17 for the Humber sets;
   - A7 upload, if time.
4. 2026-11-02, overnight GPU:
   - A18 touch scorer retrain;
   - A9 1080p ball detector retrain (owner's DA boxes, if done).
5. 2026-11-03/04: A13 dry run, fix what breaks. 2026-11-05: freeze; no new features.
6. After Ottawa: the rest of Phase B (B2-B4 and B8 were done 2026-10-31), with Phase C alongside (C1 first, so new gyms stop needing labels).
7. Then Phase D in the order that matters to the owner's coaching.

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
