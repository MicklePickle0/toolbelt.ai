# To do

Ordered by priority within each section. Scores are from the labelled `titan-rise_set1`
(touches from models that never saw the rally scored). Last updated 2026-10-26, merging
the desktop and laptop conversations (see CLAUDE.md, "Working across machines").

## Now

- [x] **Label `mac-chrome_set1`** (owner): done, 39 rallies, 246 touches, 20 pass scores.
      Two labelled sets plateau the touch scorer (~80% found, 88-92% real, team 90%);
      more full Humber sets add little. Next labels: ~10-15 rallies from a NEW venue (the
      honest test, since both labelled sets are now in training), and targeted rare events
      (blocks, free balls, dumps, replays). Then corrections from the review UI.
- [ ] **Run the remaining 10 videos** (owner): `.\scripts\process_videos.ps1`. Resumes if
      stopped, and runs `find_play` first so ball detection skips the time between rallies.
      Was ~1.6 h of ball detection per set before play spans and ball following; the new
      time is still to be measured (see "Detection speed" below).
- [x] Touch scorer retrained on both labelled sets (`titan-rise_set1` + `mac-chrome_set1`).
      Cross-court: trained on one, tested on the other: 70-83% of touches found, 86%
      real, team 83-85%. `titan-rise_set1` is the manifest's test split, so the next honest
      test is a labelled set from a new venue.

## Detection speed (on the desktop, with footage)

- [ ] **Check play spans on the labelled sets**: `find_play.py --video titan-rise_set1`
      and `--video mac-chrome_set1`. Every labelled rally must be covered; note the
      share of the video kept. Tune `activity.play:` if a serve is cut.
      *Expect a small saving:* on the desktop (2026-10-26), play windows from motion
      kept 85-96% of the video (players' speed) or 89% (the frame activity score, its 50th
      percentile) before any labelled rally was cut: players walk, rotate and celebrate
      between points about as much as they move in play. Rally time itself is only ~28% of
      a set. If `find_play` also keeps ~90%, it may not be worth its risk.
- [ ] **Two versions of the same idea exist; keep one.** Ball following here
      (`detect/follow.py`, laptop) and the near-the-ball crop on branch `gpu-play-windows`
      (`detect/roi.py`, `detect_ball.py --roi`, desktop). The crop was measured on the
      desktop: simulated on both labelled sets with `scripts/simulate_roi.py` (replays saved
      full-search detections, no GPU), ~2.6-2.7x fewer detector passes with touches and
      rallies unchanged (titan-rise 39 -> 38/42 rallies); and with the real detector on 40 s
      of titan-rise: 5.3 -> 12.7 frames/s (2.4x), and of 894 frames where the full search
      was sure of the ball, the crop found it in 886 (99%), 97% at the same spot. The limit
      is the ~40% of frames with no ball (full scan). Compare `follow.py` on the same 40 s
      and keep the better one.
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

- [ ] **Blocks**: 8 of 18 found, and behind most wrong point outcomes. Ball near the net
      tape, deflection is small and often hidden.
- [ ] **Inferred serves**: right about half the time (sent to review). Mostly from touches
      given the wrong team (team right 90%).
- [x] **Rallies built from touches** (`rally/segment.py`, `segment_from_touches`), with a
      missed opponent serve inferred from the rally (2+ touches by one side first), groups
      with no serve or exchange dropped, and other courts' balls (outside our airspace)
      ignored. Out-of-fold on titan-rise: rallies 32 -> 39/42, false rallies 8 -> 0,
      serving team 20 -> 34. Inferred serves are right only ~half the time (lower
      confidence: review queue).
- [ ] **Our serve from how the ball enters the frame** (branch `our-serve-entry`, not
      merged): our server stands behind the camera, so the ball appears big at the top of
      the frame. Big ball at the top, after 3 s without a touch, far away within 2 s: 28 of
      33 of our serves, 4 false alarms over both sets (today's touches: 22/33). Rallies
      found 39 -> 42/42 (titan-rise) and 37 -> 39/39 (mac-chrome). Depends on the camera
      being behind our end line (check on new angles). Their serves are fine (44/48).
      Placed at the hit, not the toss (where the ball stops rising): none of 29 now 0.4 s+
      early (was 10); the hit touch is no longer an extra "dig".
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

- [ ] **Pass 0-4 from the setter's feet** (branch `pass-scoring`, not merged): the ball's
      3D depth is too weak (perfect passes came out 8 m from the net); the feet of the
      player who plays the pass track the owner's scores (median distance from the target
      0.9 / 1.9 / 3.1 / 4.6 m for 4 / 3 / 2 / 1; 40 labelled passes). Draft config grades:
      51% exact, 90% within 1. Distance-only limits tuned leave-one-out: 69% / 95%
      (4 up to 1.5 m, 3 up to 1.75 m, 2 up to 5.25 m): the owner decides the thresholds.
      `scripts/eval_passes.py`.

## Milestone 9: stats and export

- [x] Stat sheet, team and per player (`scripts/stat_sheet.py`): credits follow the
      rally outcome and the owner's counting rules.
- [ ] **Clips per stat** (branch `clip-export`, on `stat-sheet`, not merged):
      `scripts/clips.py --stat kills [--player 13]` cuts clips plus a highlight reel.
- [ ] Combos (setter-to-hitter kill %), DataVolley export.
- [ ] Before the stats export: confirm how scrappy non-setter 2nd touches are counted.

## Milestones 10-12 (added to CLAUDE.md 2026-10-26)

- [ ] 10. Setting arc analysis (release, apex, tempo, landing vs antennas; per setter and
      set type; overlay and side-view diagram).
- [ ] 11. Serve-receive by zone (per player: contact point, zone 1-6, pass score map).
- [ ] 12. Serve placement on the opponent side (zones, landing points with confidence).

## Branches not merged yet (all pass their tests)

Built on the desktop on 2026-10-26 off `live-ball-review`; this branch already contains
`live-ball-review`, `stat-sheet` and `review-training`. Still separate: `our-serve-entry`,
`gpu-play-windows` (see "Detection speed": one of it and `follow.py` to keep),
`clip-export` (on `stat-sheet`), `pass-scoring`. Merge the ones the owner wants.

## Smaller known issues

- [ ] `mac-chrome_set2`: a camera jump in the last ~25 s can't be fixed (no saved frame
      after it).
- [ ] `mac-iron_set1`, `reach-apex_set2`: small leftover camera offsets (20-40 px in 4K).
- [ ] A line judge on our sideline can get a green ring.
- [ ] Far-side and floor positions are less precise in depth.
