# To do

Ordered by priority within each section. Scores are from the labelled `titan-rise_set1`
(touches from models that never saw the rally scored). Last updated 2026-10-25.

## Now

- [ ] **Label `mac-chrome_set1`** (owner). A train-split set on a different court: improves
      touches, actions and outcomes, and gives a fair test on footage the scorer never saw.
- [ ] **Run the remaining 10 videos** (owner): `.\scripts\process_videos.ps1`. About 15-20 h
      of GPU in total; resumes if stopped.
- [ ] **Retrain the touch scorer on train-split sets** once `mac-chrome_set1` is labelled,
      and keep `titan-rise_set1` (test split) only for testing.

## Milestone 4: touches and rallies

- [ ] **Blocks**: 8 of 18 found, and behind most wrong point outcomes. Ball near the net
      tape, deflection is small and often hidden.
- [ ] **Inferred serves**: right about half the time (sent to review). Mostly from touches
      given the wrong team (team right 90%).
- [ ] **Serve-only rallies from our off-screen server** (aces, service errors): 3 of 42
      missed.
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
      replay / not a rally, suggested outcome, serving team, rally end. Saves to the event
      DB, logs every change in `corrections`; re-running the pipeline keeps reviewed rallies.
- [ ] Use reviewed rallies as training data: export them like `events.csv` and train the
      touch scorer and action rules on them too.
- [ ] Keyboard shortcuts (Streamlit has none built in), and the clip with the overlay drawn.
- [ ] Fix the player on a touch (needs identity, Milestone 7); one fix relabels the
      whole track.

## Milestone 7: player identity

- [ ] Roster and lineup entry, jersey-number reading, rotation tracking. Also re-links
      player tracks that break between points.

## Milestone 8: pass scoring

- [ ] Serve-receive pass 0-4 from where the pass goes, checked against the owner's pass
      scores (20 labelled so far). Thresholds in `config/pass_scale.yaml` still to confirm.

## Milestone 9: stats and export

- [ ] Stat queries (per player, combos like setter-to-hitter kill %), DataVolley export.
- [ ] Before the stats export: confirm how scrappy non-setter 2nd touches are counted.

## Smaller known issues

- [ ] `mac-chrome_set2`: a camera jump in the last ~25 s can't be fixed (no saved frame
      after it).
- [ ] `mac-iron_set1`, `reach-apex_set2`: small leftover camera offsets (20-40 px in 4K).
- [ ] A line judge on our sideline can get a green ring.
- [ ] Far-side and floor positions are less precise in depth.
