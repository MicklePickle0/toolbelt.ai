# To do

Ordered by priority within each section. Scores are from the labelled `titan-rise_set1`
(touches from models that never saw the rally scored). Last updated 2026-11-16; first written 2026-10-26, merging
the desktop and laptop conversations (see CLAUDE.md, "Working across machines").

## Fall Classic, Ottawa, 2026-11-06/07 (deadline)

Owner has only a laptop + phone there; 6-9 sets on day 1, stats wanted by end of day.
Plan (2026-10-28): record each set on the phone at 1080p60, upload the file after the set
(not a live stream: gym upload is unreliable and the pipeline needs whole rallies), process
it as a batch job, look at results in a browser. Live AI stats: not feasible (out of scope).
Cost if cloud: ~$5-10/day on AWS (g5/g6 ~$0.8-1/h). Home PC first, cloud as a backup.

**Revised 2026-11-01: the order and status are in CLAUDE.md's Roadmap.**
- Done 2026-10-31/11-01:
  - pass score and jersey in review (A16);
  - pipeline pass scores (B8);
  - jerseys by PARSeq plus pose (B2-B4);
  - our server from who walks into view;
  - the owner's pass rules (low pass is a 1; overpass rules in CLAUDE.md);
  - web folders, Search, reels and game summary.
- Measured on the owner's `pakmen-black_set2` labels: jerseys 47% named, 92% right; pass scores 69% exact, 92% within 1. Attacks named: 3 of 24 (open, B5).
- **`pakmen-gold_set1`, first new-gym test (owner reviewed all 47 rallies, 2026-11-02; exported to `data/labels/pakmen-gold_set1/events.csv`; `eval_contacts.py` + `eval_review_labels.py`).** The truth is the owner's correction of the pipeline's proposals, so a touch both missed is not counted: recall is an upper bound.
  - Rallies: 43/46 found (6, 25 and 46 added by the owner), 4 false. Score in the DB 21-24; the real score was 21-25, so one point is not in the DB.
  - Touches: recall 87%, precision 89% (346 labelled, 341 predicted); team right 86%; action right on found touches 64%.
  - Missed most: blocks 6/18 found (10 of ours missed), our attacks 9 missed, their serves 6 missed. False touches: 39, mostly our set/dig/attack.
  - Action mix-ups (owner -> pipeline): dig as set 8, set as free ball 7, set as attack 6, dig as attack 5, dig as overpass 4.
  - Point winner from who serves next: 35/42. Point outcome (the how): 18/42; 13 of the owner's 23 kills came out as other, ball-handling error or reception error.
  - Jerseys: 45% of our touches named, 88% right (94% at confidence >= 0.6, 66 named).
  - Pass scores: owner scored 20 of 22 receptions, average 2.8 (6x4, 6x3, 6x2, 2x1). The pipeline scored only 8 of them (75% exact, 88% within 1), averaging 3.38 where the owner gave 3.00: too few and too high, as on `pakmen-black_set2` (issues-log "Still open").
  - **Errors cascade (measured 2026-11-02):** a rally where every touch was found with the right team is "clean". Clean rallies are rare: 8/46 (gold set 1), 9/45 (black set 2), 8/42 (titan-rise). Actions right on found touches, clean vs not: 94% vs 67% (gold), 80% vs 62% (black set 2), 89% vs 80% (titan-rise). So the touch-order rules are mostly right when their inputs are; one missed touch or wrong team shifts the place of every touch after it in that side's run, and the outcome follows the last touch.
- Next for Claude: A15 rally-step fixes, A11 review shortcuts, A17 for the Humber sets (pose_people + find_rallies).
- Owner, 2026-11-01: recording for Ottawa changes to 4K, keeping the 4K files and processing a 1080p60 copy. The third-set switch (A10) is deferred.
- Processed Durham Attack sets: `pakmen-black_set1` (14/53 rallies reviewed) and `pakmen-black_set2` (46/46), both re-run with jerseys and pass scores (event DB schema v8).
- Calibrated, not processed: `416-knight_set2/3`, `mac-platinum_set2`. `pakmen-gold_set1/2` processed 2026-11-01/02; set 1 reviewed (47/47), its mistaken rally 48 deleted at the owner's request (DB backup `data/setter_eyes.db.backup_2026-11-02_before_gold_rally48`).
- Not calibrated: `416-knight_set1`, `mac-platinum_set1`, `pakmen-black_set3` (side switch).
- Every other processed set still has the ball path from before the poster fix (A17).

- [x] **Speed-ups, 2026-10-29: ~2 h -> ~33 min per 20-min 1080p set.** Measured on a 1080p
      copy of titan-rise_set1 (`titan-rise_set1_1080p`, same frames, labels apply; 18.4 min):
      find_play 68 s, ball 21 min (25.9 frames/s), people 5.8 min, track_ball 68 s,
      track_people 19 s, rallies 3 s; camera tracking now runs alongside the detectors.
      1. Ball tiles by video width (`detect.ball_tiling`): 1080p 2x2 at 960 (3.6x; 93.5%
         of the 3x3@1280 sure balls on pakmen-black_set1); 4K unchanged.
      2. People at 10 Hz (`detect.person_sample_hz`), filled in to 30 Hz by track_people
         (`sparse_detection_cfg`); touch scorer retrained on 10 Hz tracks. OOF on both
         labelled sets: within noise (titan actions 73 -> 71%, mac 78 -> 79%).
         Lower person input size: 98-99% of people but only ~15% faster, not used.
      3. track_camera alongside detection in the app queue (`app.background_steps`).
      Accuracy, scorer trained on mac-chrome only, titan 4K -> 1080p fast: touches found
      83 -> 79%, real 92 -> 94%, actions 75 -> 69%, rallies 41 -> 38/42 (false 5 -> 2),
      winner 35/41 -> 31/38, outcome (inferred winner) 13/41 -> 16/38. Part of the loss is
      1080p itself (not separated). Next: retrain the ball detector at 1080p scale (labels
      downscaled, no new labelling), then a smaller model / TensorRT if needed.
- [ ] **P0 Speed: ~30 min per 1080p set.** (superseded by the entry above; history:) Measured 2026-10-28, `pakmen-black_set1` 5:00-7:00,
      no play spans: ball detection with following 8.6 frames/s = 452 s for 120 s of video
      (3.8x real time, ~75 min for a 20-min set); following only cut passes 9 -> 6 per frame
      (62% of frames scanned in full: the detector is unsure at this new gym). Full search
      6.1 frames/s (604 s, ~100 min a set); people 29.2 frames/s (135 s, ~22 min a set).
      So ~100 min a set today, ~3x off the target.
      People at 10 Hz, TensorRT/fp16, fewer/larger tiles at 1080p (may need retraining at
      DA scale), play spans.
- [ ] **P0 Web app on the desktop, reachable from the laptop/phone** (Tailscale, owner
      installs; PC set to never sleep, auto-start).
      - [x] Skeleton (2026-10-28): `scripts/serve_app.py`, FastAPI (`src/setter_eyes/api/`)
            + a plain JS page (no build step). Videos page (calibration, caches, job
            state), processing queue (`api/jobs.py`: one video at a time, logs, cancel,
            resume; stops before `track_ball` when there is no calibration, so detection
            can run while the owner calibrates), rally review (same saving as the
            Streamlit UI, review queue first, clip with frame stepping, add touch at the
            current frame), stat sheet (any sets combined, pipeline or hand labels).
            Checked on the real DB at desktop width; phone layout still to check on a phone.
      - [ ] Upload a video from the phone, or a YouTube/Drive link (yt-dlp), into the manifest.
      - [x] Progress in the queue (2026-10-28): bars for the whole job (steps weighted by
            `app.step_weights`) and the running step (parsed from its "n/total" lines,
            with minutes left); the log refreshes live with a text bar on top. Detection
            prints a progress line at least every 20 s, track_camera as frame n/total.
      - [ ] Measured: `pakmen-black_set1` (first job from the app, 2026-10-28): find_play
            kept 91% of the video; ball detection 8.5-8.8 frames/s (~65 min for the set).
      - [x] Browser calibration (2026-10-28, owner: don't expect auto line detection on
            high-school gym floors). Calibrate page per video: pick a frame (exact, via
            ffmpeg seek, ~0.7 s at 1080p), press-and-drag with a magnifier to place the
            court points (skip hidden ones), then points along lines, check the court
            drawn back over the frame, save. Same solve as calibrate.py's click UI
            (`calibration/solve.py`). Previous calibrations kept in
            `data/calibration/history/`. Processing can start before calibrating: the job
            stops at `track_camera` (now in `app.steps`; it was missing, new videos need
            it) and is resumed from the page. Automatic fits no longer count as calibrated
            (`app.accept_auto_calibration: false`), which also covers A6 for now.
            Phone layout still to check on a phone.
      - [x] Split a rally in review (2026-10-28, owner: two points counted as one): play to
            the second point's serve, "Split here"; touches from there move to a new rally,
            whoever serves it won the first part (outcome re-judged), both go back to the
            review queue, logged in `corrections`. Web app only (not the Streamlit review).
      - [ ] Clips per stat; one-key point us/them on the review page.
      - [x] Passing average on a 4- or 3-point scale (a 4 counts as 3), picked on the
            stats page; passes are always stored 0-4.
      - [ ] Pass average in the stat sheet: pipeline rallies have no pass scores yet
            (only aces count, as 0), so it reads 0.0 until passes are scored.
- [x] **B1 Rosters** (2026-10-28): several rosters (number, name, height in cm or 6'2",
      position, libero), each applied to the videos it played, on the app's Rosters page.
      Files in git (`data/rosters/<slug>.yaml`, `assignments.yaml`), copied into the event
      DB (schema v6: `teams.roster_slug`, `players.height_cm` / `position`) so they can be
      queried; touches credited to a jersey follow when a video's roster changes.
- [ ] **P0 Calibration confidence**: auto calibration gave 0.87 on a wrong fit (far
      wall/net) at the DA gyms; make the confidence honest so the app asks for taps.
- [ ] **P0 DA footage processed end to end** (10 of 12 sets downloaded, 1080p59.94;
      Unity Force sets bot-blocked by YouTube, retry). Needs manual calibrations first.
- [ ] **P1 DA labels** (owner): ~80 ball frames + ~10-20 rallies from 2 DA gyms, then
      retrain the ball detector and touch scorer; this is also the new-venue test.
- [ ] **P1 Jersey numbers -> per-player stats** (Milestone 7, started 2026-10-28).
      Findings on `titan-rise_set1` (4K): back numbers are large and readable by eye in
      most crops of near-side players (boxes 350-650 px tall); the libero's green jersey
      stands out. Two problems:
      1. *Reading:* off-the-shelf EasyOCR reads ~6/40 crops (the outlined font defeats its
         text finder); its recognizer alone gives noise. Next: a stronger pretrained
         reader (PARSeq, via torch hub: needs the owner's OK to run downloaded code) or
         our own small digit classifier trained on crops the owner tags in the app
         (~300-500 crops, ~1 s each), plus votes per track.
      2. *Who touched:* in ~1/3 of 30 labelled touches the toucher's track shows a
         different player (e.g. a "#12 receive" crop shows #1). Pick the toucher better
         (hands/pose at the ball, not the nearest feet) before numbers can help.
      Roster in the manifest is still empty (`roster: []`): owner to add numbers + names.
      (Since done: the Venom Riot roster is in `data/rosters/venom-riot.yaml` and assigned to every video.)
      **1080p check, 2026-10-31:**
      - *Player size:* near-side players are a median 262-314 px tall at 1080p (titan 1080p copy, both Pakmen Black corners; 523 px at 4K), so a back number is ~35-40 px tall.
      - *Readable by eye:* most back views in contact sheets of `pakmen-black_set2` and `titan-rise_set1_1080p` (12, 13, 14, 17, 23, 5, 8). Front views, side views and players behind teammates are not.
      - *Crop filter:* about half the "near-side" boxes were coaches, spectators and bench players along the sideline, so crops must come from player roles only.
      - *The roster helps:* it is a fixed set of ~12 numbers, so reading is a choice among the roster numbers, not free OCR.
- [x] **Jersey numbers + who touched it, built 2026-10-31** (owner OK'd PARSeq). New app step `pose_people` after `track_people`.
      - *Pose:* RTMPose on everyone on our court at that moment (`ours_now`; a track's lifetime role missed the setter and hitters at the net), 10 Hz, batched on the GPU: ~4-5 min a 20-min 1080p set.
      - *Numbers:* back views read by PARSeq, limited to the roster.
      - *Who touched it:* `find_rallies.py` picks the toucher by forearms/hands at the ball (`identity/toucher.py`). The number comes from that track's reads within +-4 s (`identity/numbers.py` NumberIndex); the tracker passes ids between players. A number another track shows more strongly then loses confidence (one player per number).
      - *Measured on titan-rise_set1_1080p* (`eval_players.py`, 115 labelled touches of ours with a jersey):
        - at the labelled moments: 67% named, 83% right; confidence >= 0.5: 54% named, 92% right;
        - through the pipeline's own touches: 50% named, 91% right; the rest left for review.
      - *Error split:* toucher wrong 7, the track's number wrong/missing then 37, nobody posed near 4.
      - mac-chrome_set1 (4K, out-of-fold touches): at the labelled moments 62% named, 75% right; through the pipeline 44% named, 79% right. At confidence >= 0.5 (now `identity.min_player_confidence`): titan 1080p 43% named, 90% right; mac 34% named, 84% right.
      - Owner reviews the unnamed touches (Player box in review).
      - titan-rise_set1 at 4K, same video (out-of-fold touches, confidence >= 0.5): 41% of our touches named, 96% right (1080p: 43%, 90%).
      - pakmen-black_set2 by eye (40 named touches): numbers right wherever visible, except the libero: the white #6's number reads poorly, so the libero's touches got other numbers (3 of 40).
        Fixed with the jersey colour (LiberoColour): it learns the libero's and the team's torso colour from the clear reads. The libero's named touches went 8 -> 17, the 3 errors became #6, and no team-coloured player got #6.
        It only switches on when the colours clearly differ: Humber's dark-green libero vs black kit doesn't (lightness 88 vs 82), so it stays off there.
      - Durham Attack by eye (Claude, pakmen-black_set2, 40 named touches spread over the set, after the libero fix): the number visible on the picked player in 26 of 40, all 26 right; 14 couldn't be judged from the crop (front or side on). Not owner labels: the owner's ticks in review will measure it properly.
      - **Durham Attack, owner's labels (pakmen-black_set2, confirmed 2026-11-01: players on 127 of our 130 touches, 25 pass scores):** pipeline jerseys (cached run from before the review) named 54/127 (43%), 49 right (91%); confidence >= 0.7: 48 named, 94% right. Wrong: #12 as #6 twice, #17 as #12, #17 as #23, #5 as #17. 58 left unnamed, 15 with no pipeline touch there. Pass scores: 13 of 25 scored, 62% exact, 85% within 1 (owner mean 3.08, pipeline 3.23); one owner 1 scored as 4.
      - **Our server (owner, 2026-11-01), `identity/server.py`:** a serve with no player named takes the number of the deepest track of ours that comes into view from the toss to 3 s after (the server walks in, back to the camera; read >= 0.9). The libero never serves, so no serve gets a libero number.
        Against no server rules, named / right: set 2 (owner's labels) 54 / 91% -> 59 / 92%; titan 4K 47 / 96% -> 48 / 96%; mac 45 / 84% -> 49 / 86%; titan 1080p 49 / 90% -> 51 / 88%. That's 11 more right, 1 more wrong.
        The owner's run idea (one server while we keep serving) only spread wrong names on every set (set 2: 91 -> 86%), so it's built but off (`identity.server.runs`). A rotation-based fallback is for later.
      - **Attacks named 3 of 24 on pakmen-black_set2 (receives 16/25, sets 14/30, serves 9/21; 2026-11-01), owner: numbers struggle when players jump.** Cause: the attacker's track begins at the approach. Outside hitters start outside the sideline (court x ~ -1.7 m by the net), and the track lasts only the 1-4 s of the attack. None of its poses is a back view (side on, arms up). The player's previous track ended 3-7 m away, 1-3 s earlier.
        Tried:
        - linking tracks within 1 s / 1.5 m and holding labels through the link (`track_links`, `hold_own_track: false`): +2 right, 0 wrong over 4 sets, kept; it doesn't reach the attacks;
        - holding the track's own reads from up to 15 s away: added wrong names, off;
        - posing and reading smaller players (`min_box_height_px` 300 -> 180, `min_crop_height_px` 80 -> 50): no change on set 2 (the attackers were posed already); titan 1080p 51 / 45 right -> 53 / 48; kept.
        Appearance re-identification, tried 2026-11-01 on set 2 against the owner's 100 labelled touches with a picked toucher. Not built in:
        - DINOv2-small embeddings of each posed crop (~3 Hz, 21.6k crops, 83 s on the GPU), gallery of 10.4k crops with a sure number, kNN: 63% right overall, 52% on the 46 touches the pipeline left unnamed, attacks 50%;
        - a classifier trained on this video's sure crops, the touch's own track held out: also 52-60% on the unnamed. Its confidence doesn't separate right from wrong (>= 0.7: 56%).
        Teammates share a kit, and a generic model mostly sees the kit. The 92%-right output would suffer, so it's not built in.
        - Elimination (players read on court in the last 60-120 s, minus the numbers other tracks show then, minus the libero for attacks): never down to one; 3-5 candidates remain, because only ~2 other tracks have a sure number at any moment. 21 of the 67 unnamed touches have no toucher found (nobody's hands at the ball).
        - Speed-gated links (as far as a player runs in the gap, up to 3 s and 4-6 m/s, only when it's the one candidate both ways): no change on any set; there's never a single candidate that far.
        Left to try: a sports-trained re-ID model (OSNet-style, trained on players), 4K crops, or the owner's review filling attacks. Scripts: scratchpad `reid_try.py` / `reid_probe.py` (not kept).
      - Durham Attack by eye, random sample (Claude): 40 of our touches from the owner's reviewed rallies on pakmen-black_set2, named or not. The pipeline named 19: 11 could be checked by eye, all right; 8 couldn't be judged. Of the 21 left unnamed, several were clear (the libero's receives, #17's sets), so it errs toward leaving touches for review.
      - Track switches as identity errors, labelled touch pairs on one identity that were two real players (`eval_track_switches.py`): tracker ids 44/78 (56%) -> after binding numbers 14/41 (34%) -> as named in the output (confidence >= 0.5) 7/30 (23%), over titan 1080p / 4K / mac (1/7, 2/10, 4/13). The rest are misreads giving both players one number.
      - Later (after Ottawa): fine-tune PARSeq on crops labelled by the owner's review ticks, or by a local vision-language model as a teacher (Qwen2-VL, Florence-2 on the 4070), for this font and the libero's kit. Not a cloud model on every crop: too slow and costly, and the players are minors.
      - Track switches (`eval_track_switches.py`): 49-62 tracks a set (about 1 in 5) carry 2+ players' numbers. On labelled touches where the track had changed hands (titan 1080p / 4K / mac: 17 / 14 / 18), the number read around the touch was right 14 / 8 / 13 times, one number for the whole track 1 / 3 / 1.
      - In the event DB (2026-10-31): pakmen-black_set1/2 re-run with `pose_people` + `find_rallies`. Every reviewed rally, touch and correction was kept (DB backed up first: `data/setter_eyes.db.backup_2026-10-31_before_jerseys`).
        Guesses were added to reviewed touches that had none (`db/pipeline.py` `_guess_on_reviewed`, confidence < 1): set 1 has 13 pass scores and 61 players, set 2 has 14 and 56.
        Stats: team pass average 2.87 over 23 passes (it was 0, from aces only); the libero #6 leads with 9 passes at 3.22.
        Side effect: set 2 gained 4 unreviewed pipeline rallies, the known false ones (0:32, 7:27, 8:56, 14:12, A15); owner: dismiss them as not a rally.
      - Speed: pose_people 4.5 min (1080p), 8.6 min (4K, decoding); GPU ~15% busy, so the CPU (decode, crops) is the limit.
- [ ] **Pass scores at 1080p, 2026-10-31** (`eval_passes.py`, labelled touches, setter's feet):
      - titan-rise_set1 4K: exact 56%, within 1 83%;
      - the same video at 1080p: exact 58%, within 1 79%;
      - mac-chrome_set1: exact 60%, within 1 100%.
      So 1080p doesn't hurt. The big misses (a 4 scored as 1) come from the wrong next player or the wrong touch.
      **Done 2026-10-31:** pass scores in processing (`score_passes`, with a confidence), and pass score + player editable in review (A16). A pipeline guess counts as a label only once set or ticked (✓).
      End to end, touches from models that never saw the rally: titan 4K 17/20 scored, 53% exact, 94% within 1; mac 15/20, 53%, 93%; titan 1080p 14/20, 43%, 71%. The 1080p misses come from upstream (receives taken as our serve, or outside a rally).
      Owner: tick or set pass scores while reviewing Durham Attack sets, to tune the thresholds there.
      **2026-11-01, owner's rules:**
      - Height: a pass in a 4's spot but very low is a 1. `low_pass.max_apex_m` 2.4. Labelled apexes: 1s from 1.8 m, 2s from 2.5, 3s/4s from 2.9; they overlap above that.
        pakmen-black_set2 against the owner's 25: 62 -> 69% exact, 85 -> 92% within 1. Titan and mac unchanged. The 1.8 m pass also helped set the threshold, so this isn't independent yet.
      - An overpass on serve receive is automatically a 0 (no pass buttons in review); one on a dig isn't a reception.
      - An "overpass" our side plays again is a receive, or a dig (`fix_overpasses`).
- [ ] **P1 Third-set side switch** (A10). Seen in `pakmen-black_set3` and `416-knight_set3`
      (2026-10-28): one continuous recording; at ~8 points the phone is carried to the other
      end (30-45 s of walls and floor), then a completely different view (other corner, other
      wall); our team stays on the near side. Needs a second full calibration, not a nudge.
      Found cheaply before any calibration: tiny grey thumbnails at 2 Hz, each compared with
      the first and the last minute's views: pakmen-black_set3 is view A until 4:20 (0.8-1.0
      similarity), the walk at 4:30-4:50 (~0), view B from 5:00 (0.9-1.0); ~47 s CPU for a
      10-min set. Plan: detect views in the first step, ask for one calibration per view on
      the Calibrate page (the second applies from where view B settles), treat the walk as
      no camera (no rallies there), track the camera per view.
- [ ] **P1 Manual stat tracker** in the app (iStatVball-style, works offline on the
      phone, syncs later). Owner to pick the stats.
- [ ] **P2 Cloud GPUs** (container + job runner + upload bucket, ~2-3 days): only if the
      home PC can't reach ~35 min/set by ~2026-11-02.
- [ ] **Dry run** ~2026-11-03/04: owner on the laptop away from home, full day's flow on DA
      sets.

## Web app: folders, search, reels, game summary (owner, 2026-11-01)
- [x] Sets in tournament -> game folders (`data/folders.yaml`, Edit folders on Stats; a set's game defaults to its key before _set).
- [x] Search tab: our touches by player, play, how the point ended, pass score and who set it (#17 set to #12). Each point's clip plays on the page; the ticked ones become a reel.
      Reels (`reels.py`): each clip 3 s before to 2 s after the touch, a ring round the toucher (hands at the ball, followed through the clip), all at 1280x720 30 fps, joined. Checked on 2 of #12's attacks off #17 (both ringed; during a jump the ring stays on the floor below the player).
- [x] Game summary on Stats: sets by hitter and by attack zone (ball 3D at the hit, else the hitter's feet), kills/errors; pass scores 0-4 and a court map of where each pass was played (dots coloured by score, or a heatmap). Counts as the stat sheet does (rallies with a winner; 33 passes, 2.58 on pakmen-black 1+2).
      Schema v8: `contacts.pass_x/pass_y` (pass spot), `player_x/player_y` (toucher's feet); re-runs fill them on reviewed touches too.
- [ ] Owner: put the sets in tournaments/games (Edit folders); check reels and the summary on a phone.
- [ ] Next: zones for the 18 attacks with no position (touches added in review have no pipeline touch); a reel's ring during jumps.

## pakmen-black_set2 reviewed (owner, 2026-10-29..31): what the corrections show

All 46 rallies reviewed in the web app (event DB, no events.csv; 5 missed points added, 4
merges). Pipeline output came from ONE run (2026-10-29 01:39, scorer trained on titan-rise_set1
+ mac-chrome_set1 only); nothing retrains during review, so later rallies were not helped by
earlier fixes (only `reinfer_winners` re-guesses unreviewed winners after an add/merge/split).
Measured from `corrections` (first half / second half of the set):
- Pipeline touches 114 / 103; deleted 33 / 38; added 41 / 68; action fixes 28 / 28; team
  fixes 11 / 12. Not better later on; quality follows the 3D fit (below), not the order.
- Of 78 deleted pipeline touches, 36 were re-added within 0.4 s (mostly the same time, other
  action/team: the owner's delete-and-re-add, effectively an edit); 42 were false, 23 of them
  "us attack", nearly all with no 3D and an image-only velocity change.
- 73 touches missed: their serve 11, our attack 11, our receive 8, our set 8, their attack 7,
  our block 6 (attack+block merge at 30 Hz, known).
- **Action rules are not the main problem:** in rallies whose touches and teams were all right,
  actions were 20/20; in rallies with a missed/false/wrong-team touch, 70/126.
- **This set's 3D is weak:** touches with a 3D position 30% (titan 76%, mac 70%, pakmen-black_set1
  68%, same gym); confident flights 6% vs ~20%; floor contacts 16 vs 46-62. By 2-min window
  the confident-flight share collapses at 6-8 min and 12-18 min (0-20%, flight fit error
  13-16 px) and is fine elsewhere (40-73%); set 1 stays 70-90% throughout. Cause: a ball
  printed on a wall poster (fixed, below).
- **Bug: pixel features aren't scaled for 1080p.** `contacts/features.py` passes `size_px`,
  `speed_in/out`, `vy_in/out` as raw pixels; the scorer was trained on 4K. Ball size median
  45.9 px (titan 4K) vs 24.5 (same video at 1080p) / 25.8 (pakmen-black_set2): at 1080p every
  ball looks twice as far away to the scorer (size is its depth cue).

Plan for the other DA gym 1 sets (pakmen-black 1/3, mac-platinum 1-2, 416-knight 1-3).
Note: set 1 and set 2 are filmed from different corners of that gym.
1. [x] Set 2's flight failures (2026-10-31): a volleyball printed on a player poster on the
   wall, 42% of the set's ball detections; the tracker's whole-track "never moves" check
   let it through. Stillness now judged per second (`ball/tracker.py`, issues-log s6).
2. [x] Pixel features in 4K units (`contacts.feature_unit_px`, issues-log s8).
3. [x] `scripts/export_reviewed.py` -> `data/labels/pakmen-black_set2/events.csv` (45 of
   46 rallies; rally 16 at 7:39 has no touches). Label check warnings for the owner:
   0:10.28 rally starts with their dig (recording started mid-rally?), 0:37.24 starts with
   our attack (serve missing?), 9:05.48 four touches by us in a row, 21:00.96 our receive
   with no serve before it.
   Results on set 2 (scorer trained on titan + mac, not on set 2):
   | | before | poster fix | + 1080p fix |
   |---|---|---|---|
   | touches found / real | 76% / 78% | 87% / 88% | 89% / 87% |
   | team right | 77% | 78% | 85% |
   | actions right (found touches) | 51% | 50% | 60% |
   | rallies found / false | 40/45 / 1 | 45/45 / 2 | 44/45 / 5 (one is 7:39) |
   | winner from next serve | 34/40 | 33/45 | 39/44 |
   | outcome, true / inferred winner | 12 / 10 of 40 | 19 / 14 of 45 | 23 / 21 of 44 |
   Out-of-fold on the Humber sets with the poster fix (current code, so this is also the
   first measurement of the team decoder above), titan-rise / mac-chrome before -> after:
   found 88 -> 89% / 87 -> 86%, real 93 -> 96% / 91 -> 89%, team 86 -> 89% / 91 -> 92%,
   actions 72 -> 77% / 78 -> 78%, rallies 41 -> 42/42 / 38/39 (4 false either way),
   outcome (inferred winner) 17/41 -> 18/42 / 15 -> 18/38. The touch scorer was retrained
   on the re-tracked titan + mac (`models/contacts/touch_scorer.pkl`).
   Only titan-rise_set1(+1080p), mac-chrome_set1 and pakmen-black_set2 were re-tracked, all
   with `--no-db`; every other set still has the old ball path and event DB rows. Backups of
   the replaced cache files: `data/cache/<id>/backup_2026-10-31/`; the old scorer and DB:
   `data/backups/2026-10-31/`, `data/setter_eyes.db.backup_2026-10-31`.
4. [ ] Owner OK'd (2026-10-31), on condition that no data is deleted: DA gym 1 becomes a training
   venue (manifest pins it to `test`); the new-gym test moves to DA gyms 2/3 (pakmen-gold).
   Before writing the event DB for set 1/2 again, check that `db/pipeline.py` keeps every
   reviewed rally and its touches (back up the DB first).
5. Retrain the touch scorer with set 2; test on pakmen-black_set1's reviewed rallies.
6. Ball detector: self-labels from set 2's reviewed rallies (C1) + ~50 hand boxes from the
   failing stretches (A9); retrain at 1080p.
7. Re-run DA gym 1, review one more set, retrain, repeat (the scorer only learns between sets).

## Owner's sanity rules (2026-10-28)

Done (out-of-fold on both labelled sets, before -> after: titan-rise actions 73% -> 73%,
outcome with the true winner 23 -> 21/41 (2 are the convention change below), inferred
20 -> 18; mac-chrome actions 79% -> 78%, false rallies 5 -> 4, winner 32 -> 33/38,
outcome 25 -> 24 and 19 -> 19):
- [x] Missed points in review: "Missed point" box (mark start/end in a widened clip, add
      it; winners around it re-inferred), split now applies the start/end edited on the
      page first, and a > 20 s gap before a rally is shown there. Fixes Pakmen Black set
      1's missing rally between 5 and 6 (see the owner's note: the rally can be added now).
- [x] Rallies < 4 s apart merged (`rally.merge_gap_s`; labelled points are 10.6+ s apart),
      unless the later one starts with a sure serve; a no-serve piece right before a serve
      is dropped as downtime. A > 20 s gap lowers the next rally's confidence (review).
- [x] A lofted single touch sent straight back is an overpass (first touch only).
- [x] Kill when the other side can't control the ball we sent over (attack, free ball,
      overpass, dump) on its first touch (`outcome.rally_outcome`); credited to that touch.
      **Labels disagree in 4 places** (titan-rise / mac-chrome: attack or dump -> one failed
      dig labelled ball_handling_error, 3 others labelled kill): relabel to the new rule.
- [x] Assists: our set right before our killing attack.
- [x] Block from the landing point: an attack whose ball next lands on the attacker's own
      side, 1 m+ from the net (closer: a net ball), gets a block by the other side
      (`contacts.landing_block`, confidence 0.5, review). 1 found on mac-chrome, 0 on titan.
- [x] Abrupt ends in review: "Ended abruptly?" (net touch, centre line, back-row attack,
      libero set, lift, double, four touches, antenna, foot fault, rotation, injury,
      referee); `rallies.end_detail`, schema v7.

- [x] Hidden touches from the timing (owner, 2026-10-29: the far receive is often missed
      at the Durham Attack corner angle, so their set looked like the receive; a set hidden
      behind the net still happened). Labelled: serve -> receive 0.75-1.50 s, two touches
      of one side in a row at most 2.17 s apart. A receiving side's first touch > 1.6 s
      after the serve (seen, or traced back from the flight) is its set (> 3.5 s: attack),
      and the missed receive is added as an inferred touch; a gap > 2.3 s inside a side's
      run adds the hidden touch (usually the set). Inferred touches: confidence 0.3
      (review), `contacts[].inferred`. OOF: titan actions 71% (same), mac 79 -> 81% (both
      "our set called a receive" fixed). pakmen-black_set1: 3 of 29 receiving sides
      shifted, 6 hidden touches added. Tried and dropped: "a 2nd touch going over is never
      a set" (cost 4-6 sets: usually the attack after it was the one missed).
      Root cause stays: far-side touches go undetected at corner angles -> DA labels (A9).
Already in before (unchanged): receive only straight after the serve (dig otherwise,
coverage after a block); one serve per rally; winner from who serves next; inferred serve
when a receive starts the rally; stuff blocks at 60 Hz for point-ending attacks.

- [ ] **At most 3 touches a side in a row, plus a block** (owner, 2026-10-29; built on the
      laptop, NOT yet measured). In the event DB, 91 of 581 pipeline rallies (16%) had one
      side touching 4+ times in a row (blocks excluded; 5-12 per set, runs up to 9): a
      touch given the wrong team, so the ball "never crossed". The old prior only halved a
      5th touch's confidence. Now `_apply_touch_prior` decodes each rally's teams as a
      whole (Viterbi over the scorer's `p_ours`, kept on `Contact.p_ours`): a 4th touch in
      a row costs `contacts.team_decoder.fourth_touch_cost` (a block first), a 5th is
      impossible, and the serving side can't touch right after its serve; a flipped touch
      keeps `over_limit_penalty` of its confidence (review). **On the desktop:**
      `find_rallies.py` + `eval_contacts.py --oof` on titan-rise_set1 and mac-chrome_set1
      (team right, actions, rallies, winners, outcomes before -> after), tune the cost,
      then re-run the rally step on every processed set.

Still to do:
- [ ] Missed points, pipeline side (on the desktop, GPU): for gaps > 20 s, re-detect the
      ball there with a full search and a lower threshold, and keep a lone opponent serve
      that dies (serve into the net) as a low-confidence rally instead of dropping it.
- [ ] Attack only for spikes, free ball for a volley/bump over: needs pose (arm up and
      swinging at the touch, RTMPose); today it is the ball's launch (lofted = free ball).
- [ ] Two-touch side that sends it over: dump (setter) vs free ball vs set-on-1/attack-on-2
      (then the 1st touch is a set, and an assist on a kill). Needs pose / identity; today
      runs are counted from the end because ~1 in 8 touches is missed.
- [ ] Chain check over the whole rally (serve -> receive/set/attack ..., block, dig ...)
      as a constrained decoder over the classifier's action probabilities.
- [ ] Touch type from where the ball meets the pose (hands above head = set/attack,
      forearms = dig/receive): costs pose on every touch, worth measuring.
- [ ] Block-outs (tooled): already a kill for the attacker by the winner rule; the touch
      itself isn't marked.

## Owner's priorities after Ottawa (2026-11-16)

Goal: hands-free. Most rallies and stats right without review; review only what the
pipeline flags as unsure. Order: accuracy first, then the cloud (owner's choice).
- [ ] **Score and stat sanity checks** (extends A15's impossible-score check, not started).
      Sets end at 25 (third set 15), win by 2, no cap; settings, not hardcoded, since
      formats vary by tournament. Today nothing stops a set reaching 33-17
      (`pakmen-black_set1`). Rules:
      - the inferred score must end exactly at a legal set end;
      - when the score passes a legal end, the rallies around the break are flagged;
      - when it falls short, missed rallies are likely: look for gaps with play but no rally;
      - send the least sure point winners to review first.
      Stat checks:
      - one serve per rally;
      - the serving team follows the point winners;
      - kills <= attacks;
      - at most 3 touches a side (plus a block);
      - a receive only after their serve.
      Each failed check lowers the confidence of the rallies involved, so they rise in the
      review queue, never silently "fixed".
- [ ] **Ball depth anchors** (see "Now").
- [ ] **Track linking over the whole rally, without rotation** (roadmap B5). Only rules
      that can't be wrong:
      - one number on one player at a time;
      - no faster than a sprint;
      - a number read before or after an attack carries across it.
      Rotation is NOT a hard rule (owner, 2026-11-16): a missed serve, a wrong point winner
      or an unnoticed mis-rotation would shift every later rally. At most a weak hint that
      number reads override. Works per rally, so it fits cloud processing after each set.
- [ ] **Set maps (owner: both), next to the existing pass map on Stats:**
      - where sets are played from: density per setter, not coloured by pass score (same
        spot as the pass map: the setter's feet);
      - where the set goes: the hitter's contact point, by set type (left, right, middle,
        pipe, back set), needing the ball's 3D at the hit (better after the depth anchors).
        Leads into milestone 10 (set arcs).
- [ ] **Situational stats: as much as each video can give (owner, 2026-11-16).** Example: a
      setter who sets well in system but struggles off the net. All are SQL over the event
      DB (CLAUDE.md: never re-run vision), with the counts shown next to each % (small samples).
      - Setter: where each set is played from (distance and direction from the target) and
        where it goes (zone, set type), in vs out of system; the result of the attack that
        follows (kill %, efficiency) by setter location. Later tempo and arc (milestone 10).
      - Passer: pass average and error rate by receive zone (1, 6, 5, and 2/4 when they pass
        in front); a court map of their contact points (milestone 11).
      - Hitter: efficiency (kills - errors) / attempts by attack zone (4, 3, 2, pipe/back
        row), by set type, in vs out of system, by setter.
      - Server: placement by zone (milestone 12), coarse and with confidences.
      - "In system": draft = pass 3-4 (setter near the target), a setting in config;
        owner to confirm.
      - **Data gap first:** in the DB only 479 of 3551 touches have the toucher's feet
        (`contacts.player_x`); ours: sets 118/438, attacks 86/403, receives 64/239. It is
        filled only when a toucher track is picked (pose + `TouchPicker`), and many sets
        predate the pose re-run (A17). The splits are only as good as this coverage: re-run
        the processed sets, then raise the toucher rate (track linking, depth anchors).
- [ ] **Practice mode vs game mode** (owner, 2026-11-16). Game mode is the current pipeline.
      Practice mode:
      - pass scores and set location/arc only;
      - no rallies, score, point winners or outcomes;
      - a pass is a ball coming over the net, or from a coach or a toss, then our first
        and second touches.
      Chosen per video at upload; the review and Stats pages show only what that mode makes.
- [ ] Possible: SAM 2 / SAMURAI around our attacks to name attackers (3 of 24 on
      `pakmen-black_set2`). Estimate ~3-6 GPU min a set if only ~3 s before each attack.
      Test first on set 2's 24 labelled attacks (~1 h GPU). Owner hasn't decided.
- [ ] **Cloud (D8), after the accuracy work** (owner, 2026-11-16): AWS / Azure / other, so
      not everything runs on the home GPU. Processing stays fast after each set (target
      ~10-20 min), not live (owner chose this over near-live and live, 2026-11-16).

## Now

- [ ] **Ball depth anchored at touches, bounces and the net (owner, 2026-11-16).** Why:
      from a camera ~1.7 m high behind the end line (`pakmen-black_set2`: 1.71 m, f 1534 px
      at 1080p), the ray through the ball is nearly flat, so 1 m along it changes the
      ball's height by only ~10 cm near the net; ball size is +-2 px of ~24 px (~+-1 m at
      the net). Arcs fix depth by gravity alone, which fails on short flights (sets,
      quicks, blocks). Today `ball/arcs.py` uses the floor only as a lower bound.
      Plan (CPU only, from caches):
      - Touch anchor: at a touch, the ball is at the toucher's hands, and the toucher's
        depth comes from the feet (far better than the ball's). Add it as a soft
        constraint at both ends of the flight, weighted by the toucher's confidence;
        jumping attackers use the interpolated feet (`track/people.py` apparent height).
      - Bounce anchor: height 0 at a floor contact.
      - Net anchor: a flight that crosses the net reaches the net plane above the tape
        (`games.net_height_m`).
      - Beware circularity: the toucher is partly chosen by the ball's 3D position
        (pass setter). Anchor only confident touchers, or iterate once.
      - Measure before/after on the labelled sets (titan-rise_set1, mac-chrome_set1,
        pakmen-black_set2, pakmen-gold_set1): pass spot from the ball vs the owner's
        pass scores (`eval_passes.py`; today the setter's feet are used because the ball
        put perfect passes 8 m from the net), team from the ball's side (64% from 3D
        today), attack zones, arc confidence, touches found (`eval_contacts.py`).

- [x] **Label `mac-chrome_set1`** (owner): done, 39 rallies, 246 touches, 20 pass scores.
      Two labelled sets plateau the touch scorer (~80% found, 88-92% real, team 90%);
      more full Humber sets add little. Next labels: ~10-15 rallies from a NEW venue (the
      honest test, since both labelled sets are now in training), and targeted rare events
      (blocks, free balls, dumps, replays). Then corrections from the review UI.
- [ ] **Run the remaining 3 videos** (owner): 9 of 12 are processed; `leaside-panther_set2`,
      `mac-iron_set2` and `titan-rise_set2` have only part of their cache.
      `.\scripts\process_videos.ps1`. Resumes if
      stopped, and runs `find_play` first so ball detection skips the time between rallies.
      Was ~1.6 h of ball detection per set before play spans and ball following; the new
      time is still to be measured (see "Detection speed" below).
- [x] Touch scorer retrained on both labelled sets (`titan-rise_set1` + `mac-chrome_set1`).
      Cross-court: trained on one, tested on the other: 70-83% of touches found, 86%
      real, team 83-85%. `titan-rise_set1` is the manifest's test split, so the next honest
      test is a labelled set from a new venue.

## On the laptop (no footage, no GPU, no training)

Works from code, the event DB and the labels in git. Laptop env synced 2026-10-27
(scikit-learn, trackers; extra OpenCV builds removed): 138 tests pass.

- [ ] **Player identity groundwork (Milestone 7)**, start here:
      - [ ] Roster and lineups (owner): the manifest already takes `roster:` and `lineups:`
            (`docs/labelling.md`); both are empty. Suggested from labelled servers:
            `titan-rise_set1` [18, 12, 14, 1, 13, 5], `mac-chrome_set1` [1, 10, 23, 14,
            13, 5]; #6 never serves, probably the libero. To confirm.
      - [x] Rotation state machine (2026-10-27, `identity/rotation.py`, `db/rotation.py`,
            `scripts/check_rotation.py`): rotates when we win the serve back; a known server
            anchors it (substitution into the slot, or resync). Writes `rallies.rotation`
            and the server on our unreviewed serves with `contacts.player_confidence`
            (schema v5). Labelled serving sequence alone: 32/33 servers (the miss is #10 in
            for #14 on titan-rise). Pipeline serving sequence (in-fold DB): titan-rise
            17/18, mac-chrome 8/14 (its 6 false rallies shift the rotation); confidence
            falls per uncertain side-out but doesn't separate right/wrong well yet. One
            confirmed server in the review UI re-anchors everything after it.
      - [ ] Owner: rotation numbering. Now 1 = the starting lineup; coaches and DataVolley
            usually name a rotation by the setter's position (needs the setter per set).
      - [ ] Libero swaps (back-row middle replaced; needs tracks: on the desktop).
      - [ ] Review UI: show the server and rotation; fixing the server re-runs
            `apply_rotation` so later rallies follow.
      - [ ] Number voting and track binding as pure logic, tested on synthetic reads (OCR and
            the appearance gallery need crops: on the desktop).
- [ ] **DataVolley export (Milestone 9)**: `.dvw` from `rallies` + `contacts` (scout line per
      touch, point and score lines from the stored winner); placeholder numbers until
      identity. Test on a fixture DB; open the file in a DataVolley viewer.
- [ ] **Team-level combos** as SQL: kill % by set type and by rotation (per player after
      identity).
- [ ] **Ask the owner** how scrappy non-setter 2nd touches are counted; record it in
      `docs/labelling.md`.
- [ ] **Review UI**: pre-fill the point winner from `next_serve` (one key accepts it; low
      confidence first in the queue); keyboard shortcuts (point us / them, rally ended
      here, next item). Visual check on the desktop.
- [ ] **Milestones 11-12, data-free parts**: shared court-map plotting (near zones 1-6,
      opponent zones, dots by score/outcome, heatmap toggle) and serve-landing logic
      (floor crossing or first opponent touch, coarse zones, confidence, review queue),
      tested on synthetic points and arcs. Real arcs on the desktop.
- [ ] **Docs cleanup**: here, "Clips per stat" is done (`scripts/clips.py`), the Speed section
      still describes the removed `--roi`, `our-serve-entry` is merged; in
      `docs/issues-log.md`, "Still open" predates the point-winner and stuff-block work.
- [ ] Update the git remote on both machines: the repo moved to
      `https://github.com/MicklePickle0/toolbelt.ai.git`.

Later (CLAUDE.md, "Product shape"): hosted web app and player phone app only once the local
pipeline is right; no paid cloud hosting before then.

## Detection speed (on the desktop, with footage)

- [ ] **Check play spans on the labelled sets**: `find_play.py --video titan-rise_set1`
      and `--video mac-chrome_set1`. Every labelled rally must be covered; note the
      share of the video kept. Tune `activity.play:` if a serve is cut.
      *Expect a small saving:* on the desktop (2026-10-26), play windows from motion
      kept 85-96% of the video (players' speed) or 89% (the frame activity score, its 50th
      percentile) before any labelled rally was cut: players walk, rotate and celebrate
      between points about as much as they move in play. Rally time itself is only ~28% of
      a set. If `find_play` also keeps ~90%, it may not be worth its risk.
- [x] **Two versions of the same idea; kept ball following** (2026-10-27). Same 40 s of
      titan-rise, every 2nd frame, against a full search (5.3 frames/s): ball following
      (`detect/follow.py`) 10.5 frames/s (2.0x), the match ball kept in 877/894 frames
      (98.1%), 63 of 1047 sure detections missed; the near-the-ball crop (`--roi`) 12.9
      frames/s (2.4x) but 862/894 (96.4%), 96 missed. All misses were moving balls (no
      spare balls). `--roi`, `detect/roi.py` and `simulate_roi.py` were removed (in git
      history); `predict_crop` stays.
- [ ] **Check ball following against full scans** on a labelled minute or two:
      `detect_ball.py --start 60 --seconds 120 --out A.npz` vs the same with `--no-follow
      --out B.npz`, then `track_ball.py --detections` on each and `eval_contacts.py`.
      Touches and rallies should match; note the passes per frame and the time.
- [ ] **1080p30 test**: scale `titan-rise_set1` down with ffmpeg
      (`-vf scale=1920:-2 -r 30`), add it to the manifest under a new key (copy the labels,
      frames scaled by 30/59.94), calibrate, run the pipeline and compare with
      `eval_contacts.py`. The detector was trained on 4K tiles: if recall drops, fine-tune
      on the labelled frames scaled to 1080p (no new labelling needed).
- [ ] People detection still runs on every sample; skipping downtime there too needs
      `track_people` to end tracks across skipped stretches first (as `track_ball` does).

## Milestone 4: touches and rallies

- [x] **Touch timing: snap to the sharpest change of course** (`contacts.snap_window_s`,
      2026-10-26, from the owner spotting an attack marked while the ball was still
      dropping). Every touch was placed ~5 frames early: at the last sample of the flight
      before it, not where the course changes (a frame where the ball was only bridged made
      it worse). Out-of-fold, both labelled sets: timing error median 0.083 -> 0.017 s,
      within 0.05 s 8% -> 90%; touches found 79/81% -> 87/85%, real 92/88% -> 95/92%;
      actions 66/56% -> 73/79%; point outcome 15/39 -> 23/41 and 20/37 -> 23/36; serving
      team 33 -> 40 (titan-rise); false rallies 2 -> 0 and 9 -> 5; rallies found 39 -> 41
      and 37 -> 36. Touch scorer retrained with it.

- [x] **Stuff blocks** (2026-10-27; owner: count the big blocks that win points, not
      fingertip touches; an attacker who swung and whose side scored has a kill either way).
      Why blocks were missed: a block comes 0.03-0.23 s after the attack (labelled 28/28;
      every other touch pair is >= 0.28 s apart), under what the 30 Hz ball path resolves,
      so the attack and block are one touch. At 30 Hz no turn signal separates blocked
      from unblocked attacks; re-detected at 60 Hz (one crop per frame) it is better but
      still ~14/23 blocks vs 8/61 false at best, so not used in general. Used only where it
      decides a stat: a point-ending attack the attacker lost (`contacts/blocks.py`,
      `scripts/check_blocks.py`, GPU, ~5 s per attack): a second sharp turn >= 2000 px/s
      (4K) right after it adds a block by the other side and makes it block_point.
      Labelled: both stuff blocks >= 2169, the 19 attack errors <= 1565; out-of-fold
      titan-rise 1 stuff block found (the other one's rally was cut short), 0 false over 18
      checks; outcomes 22 -> 23/41. Only 2 examples: added blocks go to review (0.6).
- [ ] Check stuff blocks on more footage (label any block points in the new venue's rallies).
- [ ] **Inferred serves**: right about half the time (sent to review). Mostly from touches
      given the wrong team (team right 90%).
- [x] **Rallies built from touches** (`rally/segment.py`, `segment_from_touches`), with a
      missed opponent serve inferred from the rally (2+ touches by one side first), groups
      with no serve or exchange dropped, and other courts' balls (outside our airspace)
      ignored. Out-of-fold on titan-rise: rallies 32 -> 39/42, false rallies 8 -> 0,
      serving team 20 -> 34. Inferred serves are right only ~half the time (lower
      confidence: review queue).
- [ ] **Our serve from how the ball enters the frame**:
      our server stands behind the camera, so the ball appears big at the top of
      the frame. Big ball at the top, after 3 s without a touch, far away within 2 s: 28 of
      33 of our serves, 4 false alarms over both sets (today's touches: 22/33). Rallies
      found 39 -> 42/42 (titan-rise) and 37 -> 39/39 (mac-chrome). Depends on the camera
      being behind our end line (check on new angles). Their serves are fine (44/48).
      Placed at the hit, not the toss (where the ball stops rising): none of 29 now 0.4 s+
      early (was 10); the hit touch is no longer an extra "dig".
      Merged 2026-10-26. Out-of-fold with it: titan-rise 41/42 rallies, 0 false, outcomes
      23 -> 22/41; mac-chrome 36 -> 38/39 rallies but 5 -> 7 false, outcomes 23/36 -> 24/38.
      The 2 new false rallies (5:37.0, 13:39.9) are 0.3 s, one touch: an entry serve with
      nothing after it. Fixed: an entry serve alone with the ball followed < 0.6 s
      (`rally.our_serve.min_flight_s`) is dropped; real aces/service errors are followed
      1.3-3.4 s. mac-chrome false rallies 7 -> 5, nothing real lost.
- [ ] **Serve-only rallies from our off-screen server** (aces, service errors): 3 of 42
      missed (the `our-serve-entry` branch addresses these).
- [x] **Live-ball check, first version** (`rally/live.py`): a trailing touch > 2 s after the
      previous one while near-side players have slowed ends the rally before it. Both
      labelled sets: 5/13 dead touches caught, 0/76 real last touches cut.
- [ ] **Dead-ball touches after the whistle: the rest of the live-ball check.** Add signals 1
      (ball leaves slowly / caught) and 3 (net shake) below; the first version uses only a
      long gap plus players slowing.
      - *Problem:* the pipeline keeps adding touches after a rally has ended: the ball is
        caught, held or tossed back after the whistle. Seen on `titan-rise_set1`: 5 of 42
        rallies, extra touch 0.5-1.8 s after the end, after kills, a ball-handling error and
        a service error. Seen in the `mac-chrome_set1` demo: a net touch ended the point, and
        a "dig" was still tracked after it. Each extra touch also makes the point outcome
        wrong, since the outcome reads the rally's last touch.
      - *Why the pipeline can't hear it:* no audio, by design. A net touch, a lift or a
        line call is only a whistle.
      - *Fix, a live-ball check on each rally's trailing touches:* accept a touch only if
        play still looks on. Signals:
        1. *Ball goes dead:* little speed after the touch, no fast flight, the ball stops in
           someone's hands. Live touches send it away with pace.
        2. *Players stop:* near-side players' combined movement drops sharply within about
           1 s of the whistle (people tracks already give this).
        3. *Net shakes:* optical flow in the net region (known per frame from the
           calibration) while the ball isn't near it, which means a player touched the net.
           This catches net touches directly.
        A trailing touch that fails ends the rally before it, at reduced confidence (review
        queue).
      - *Caveats:* only 6 labelled examples so far, so hand-set rules, not a model; check
        against the labels that no real last touches are cut. Label the end of a rally at
        the whistle (as done so far) to get more examples.
      - *Backstop:* a one-key "rally ended here" in the review UI (Milestone 6), next to the
        point-winner key; corrections become training data.
      - *2026-10-27, measured on both labelled sets (16 dead trailing touches, 50 real last
        touches):*
        - Done: a touch after a floor contact is dead (the ball can't bounce in play): 3
          dead + 1 false touch came after one, no real touch. mac-chrome outcomes 24 -> 25/38.
        - Signal 1 (ball goes dead) does not separate on this footage: ball travel/speed
          after dead touches is lower (median 645 vs 1007 px/s) but overlaps; with the
          players' slowdown added, every rule that caught 2-5 of 16 dead touches also cut
          1-7 real last touches. Players barely slow right at a dead touch (median 0.85x
          the rally's pace vs 0.95x at real last touches). Not adopted.
        - Signal 3 (net shake) not tried: needs frames decoded around each rally end, and
          there are only ~2 labelled net touches to check it against.
        - Owner (2026-10-27): often nothing visible ends a point (a small net touch, a bad
          call); play just stops and a team gets the serve. So an end with no landing seen
          is normal: those rallies get no floor bonus (lower confidence, review queue), and
          their outcome can be `other`. The next rally's serving team says who won the
          point; the pipeline doesn't decide the score (CLAUDE.md), but the review UI could
          pre-fill the winner from it (to discuss with the owner).
- [ ] **Ball-size check for other courts' balls** (a ball behind our court looks too
      small for where it seems to be). Only the airspace check is used so far.

## Milestone 5: event database, outcomes, actions

- [x] Stage-1 action rules (65% right; serves 85%, receives 81%, sets 64%, attacks 53%)
- [x] Point outcome from the last touch and the winner (39/42 on labelled touches; 19/39
      on pipeline touches)
- [x] Pipeline output written to the event DB
- [x] Stage 2, free ball: an "attack" launched upward > 4 m/s is a free ball (free balls
      0/4 -> 2/4 right, no attacks lost; 6 labelled examples, confirm on more sets)
- [ ] **Stage-2 overrides** from trajectory and pose: dump, attack vs free ball
      (arm swing), block (arms above the tape). Needs pose (RTMPose).
- [ ] **Learned action classifier** (stage 3): tried at 64%, no better than the rules on
      one set; retry with two labelled sets.

- [x] **Point winner from who serves next** (2026-10-27, owner's call: the score can be
      tracked internally to judge the HOW of a point; OCR and the review key override).
      `rally/winner.py`; stored as `point_winner_source = next_serve` with a confidence, a
      running score, and the outcome (event DB schema v4, migrated in place). Labelled:
      the next server is the winner 79/79. Pipeline, out-of-fold: winner right 38/41
      (titan-rise) and 32/38 (mac-chrome); misses come from a missed or false rally in
      between, and the set's last rally (no next serve). Outcome with the inferred winner:
      19/41 and 19/38 (with the true winner: 22/41 and 25/38).

## Milestone 6: review UI

- [x] First version (`python scripts/review.py`, Streamlit): review queue (least confident
      first), rally clip, editable touches (team, action, add, delete), point winner /
      replay / not a rally, suggested outcome, serving team, rally start and end,
      a wider clip for points cut short, merging a point the pipeline split in two. Saves to the event
      DB, logs every change in `corrections`; re-running the pipeline keeps reviewed rallies.
- [x] Reviewed rallies as training data: `train_touch_scorer.py --reviewed` (rallies with
      every touch confirmed; "not a rally" ones as negatives).
- [x] Millisecond player with frame stepping, three-decimal times; rally start editable,
      wider clip, merge with the neighbouring rally.
- [ ] Keyboard shortcuts (Streamlit has none built in), and the clip with the overlay drawn.
- [ ] Fix the player on a touch (needs identity, Milestone 7); one fix relabels the
      whole track.

## Milestone 7: player identity

- [ ] Roster and lineup entry, jersey-number reading, rotation tracking. Also re-links
      player tracks that break between points.

## Milestone 8: pass scoring

- [ ] **Pass 0-4 from the setter's feet**: the ball's
      3D depth is too weak (perfect passes came out 8 m from the net); the feet of the
      player who plays the pass track the owner's scores (median distance from the target
      0.9 / 1.9 / 3.1 / 4.6 m for 4 / 3 / 2 / 1; 40 labelled passes). The owner keeps the
      draft thresholds and treats the labels as ground truth: big gaps (1 vs 2-4) mean
      the wrong player was found. Fixed 2026-10-26: the setter is found by the ball's 3D
      position among players under the ball in the image, else by hands at the ball (a
      back-row player near the camera, e.g. the libero in front of the setter, was being
      picked). Scores 2+ apart: 9 -> 2; exact 51 -> 63%, within 1 90 -> 97%.
      `scripts/eval_passes.py`, `scripts/plot_passes.py` (map + frames).

## Milestone 9: stats and export

- [x] Stat sheet, team and per player (`scripts/stat_sheet.py`): credits follow the
      rally outcome and the owner's counting rules.
- [ ] **Clips per stat**:
      `scripts/clips.py --stat kills [--player 13]` cuts clips plus a highlight reel.
- [ ] Combos (setter-to-hitter kill %), DataVolley export.
- [ ] Before the stats export: confirm how scrappy non-setter 2nd touches are counted.

## Milestones 10-12 (added to CLAUDE.md 2026-10-26)

- [ ] 10. Setting arc analysis (release, apex, tempo, landing vs antennas; per setter and
      set type; overlay and side-view diagram).
- [ ] 11. Serve-receive by zone (per player: contact point, zone 1-6, pass score map).
- [ ] 12. Serve placement on the opponent side (zones, landing points with confidence).

## Branches

All desktop branches were merged into `live-ball-review` on 2026-10-26: `detect-speedups`
(laptop work), `stat-sheet`, `review-training`, `our-serve-entry`, `clip-export`,
`pass-scoring` and `gpu-play-windows`. Work continues from `live-ball-review`.

## Speed (GPU): the desktop's measurements

Ball detection is ~2 h of a ~2.4 h set (9 tiles x every 2nd 4K frame).
- [x] **Near-the-ball search, simulated** (`detect/roi.py`, `scripts/simulate_roi.py`, no
      GPU needed): one tile-sized crop around the ball while it is followed, all 9 tiles
      when it is lost and once a second; idle mode (4 searches a second) after 2 s without
      a ball, with backfill when it returns. Both labelled sets: **~2.6-2.7x fewer detector
      runs**, touches unchanged, rallies 39 -> 38/42 (titan-rise) and 37/39 unchanged
      (mac-chrome). Settings barely matter: the limit is the ~40% of frames with no ball.
- [x] `detect_ball.py --roi` uses it. Real check, 40 s of titan-rise: 5.3 -> 12.7
      frames/s (**2.4x**); of 894 frames where the full search was sure of the ball, the
      near-the-ball search found it in 886 (99%), at the same spot in 97%.
- [x] Ball following (`follow.py`) is the default; `--roi` measured and removed (see
      "Detection speed" above).
- [x] Play windows from player movement: **not worth it.** Keeping every labelled rally
      kept 85-96% of the video (players walk, rotate and celebrate between points).
- [ ] Efficient inference: TensorRT / fp16 compile, a smaller ball model if accuracy
      holds; 1080p input; for the product, parallel cloud GPUs and on-device downscaling.

## Smaller known issues

- [ ] `mac-chrome_set2`: a camera jump in the last ~25 s can't be fixed (no saved frame
      after it).
- [ ] `mac-iron_set1`, `reach-apex_set2`: small leftover camera offsets (20-40 px in 4K).
- [ ] A line judge on our sideline can get a green ring.
- [ ] Far-side and floor positions are less precise in depth.

## Later (owner, 2026-11-16: good enough for now)

- [ ] Multi-frame ball detector (TrackNetV3 / WASB style: ~3 frames in, heatmap out):
      better on motion blur and half-hidden balls, and TrackNetV3 fills in hidden stretches
      of the path. Current single-frame RF-DETR is good enough for now.
