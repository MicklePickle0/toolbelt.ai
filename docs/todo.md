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
- [ ] **Ball-size check for other courts' balls** (a ball behind our court looks too
      small for where it seems to be). Only the airspace check is used so far.

## Milestone 5: event database, outcomes, actions

- [x] Stage-1 action rules (65% right; serves 85%, receives 81%, sets 64%, attacks 53%)
- [x] Point outcome from the last touch and the winner (39/42 on labelled touches; 19/39
      on pipeline touches)
- [x] Pipeline output written to the event DB
- [ ] **Stage-2 overrides** from trajectory and pose: dump, free ball, attack vs free ball
      (arm swing), block (arms above the tape). Needs pose (RTMPose).
- [ ] **Learned action classifier** (stage 3): tried at 64%, no better than the rules on
      one set; retry with two labelled sets.

## Milestone 6: review UI

- [ ] Scrub the timeline, fix a touch / action / player, set the point winner, resolve the
      review queue. Corrections stored as training data.

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
