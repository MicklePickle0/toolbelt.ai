# To do

Ordered by priority within each section. Scores are from the labelled `titan-rise_set1`
(touches from models that never saw the rally scored). Last updated 2026-10-26, merging
the desktop and laptop conversations (see CLAUDE.md, "Working across machines").

## Fall Classic, Ottawa, 2026-11-06/07 (deadline)

Owner has only a laptop + phone there; 6-9 sets on day 1, stats wanted by end of day.
Plan (2026-10-28): record each set on the phone at 1080p60, upload the file after the set
(not a live stream: gym upload is unreliable and the pipeline needs whole rallies), process
it as a batch job, look at results in a browser. Live AI stats: not feasible (out of scope).
Cost if cloud: ~$5-10/day on AWS (g5/g6 ~$0.8-1/h). Home PC first, cloud as a backup.

- [ ] **P0 Speed: ~30 min per 1080p set.** Measured 2026-10-28, `pakmen-black_set1` 5:00-7:00,
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

## Now

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
