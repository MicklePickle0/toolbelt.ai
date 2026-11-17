"""Who served for us (owner, 2026-11-01).

From behind our end line the server is often beside or behind the camera, so the serve
touch has nobody at the ball. After the serve they walk onto the court, into view and with
their back to the camera:
  - the server is the track of ours that first appears around the serve (from the toss,
    `enter_before_s`, to `enter_window_s` after: it wasn't in view before), deep in our
    court (`min_depth_m` from the net); its number is the surest read in the seconds after
    it walks in (NumberIndex). The libero never serves.
  - a run of rallies we serve in a row has ONE server (we keep serving while we win the
    point): the run's serves vote, and the winner names every serve of the run.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from setter_eyes.track.people import PeopleTracks


def first_seen(people: PeopleTracks) -> dict[int, tuple[int, np.ndarray]]:
    """{track: (first frame, court xy there)}."""
    order = np.lexsort((people.frame, people.track_id))
    out = {}
    for r in order:
        t = int(people.track_id[r])
        if t not in out:
            out[t] = (int(people.frame[r]), people.court_xy_smooth[r])
    return out


def name_servers(rallies: list, contacts: list, people: PeopleTracks, numbers, fps: float,
                 id_cfg: dict, liberos: tuple[int, ...] = ()) -> Counter:
    """Jerseys on our serves, in place (Contact.jersey / player_confidence), kept from
    identity.min_player_confidence on. id_cfg: the `identity:` config. The libero doesn't
    serve (owner's labels: never), so a walk-in read as a libero is someone else. Returns
    counts."""
    cfg, keep = id_cfg["server"], id_cfg["min_player_confidence"]
    how = Counter()
    if numbers is None:
        return how
    seen = first_seen(people)
    for r in rallies if cfg["walk_in"] else []:
        if r.serving_team != "us" or r.serve_contact is None or r.serve_contact < 0:
            continue
        c = contacts[r.serve_contact]
        if c.jersey is not None:
            continue
        lo = c.frame - cfg["enter_before_s"] * fps
        hi = c.frame + cfg["enter_window_s"] * fps
        best = None
        # The deepest newcomer is the server (the others step into view further up).
        cands = sorted(((xy[1], tid, f0) for tid, (f0, xy) in seen.items()
                        if lo <= f0 <= hi and np.isfinite(xy).all()
                        and xy[1] >= cfg["min_depth_m"]), reverse=True)
        for _, tid, f0 in cands:
            for after in cfg["read_after_s"]:       # its back is to us as it walks in
                got = numbers.number_at(tid, int(f0 + after * fps))
                if got is not None and got[0] not in liberos and \
                        got[1] >= cfg["min_read_confidence"] and \
                        (best is None or got[1] > best[1]):
                    best = got
            if best is not None:
                break
        if best is None or best[1] * cfg["confidence"] < keep:
            how["no one walked in with a sure number"] += 1
            continue
        c.jersey, c.player_confidence = best[0], round(best[1] * cfg["confidence"], 3)
        how["server walked in"] += 1

    # One server per run of our serves.
    runs: list[list] = [[]]
    for r in sorted(rallies, key=lambda r: r.start_t):
        if r.serving_team == "us" and r.serve_contact is not None and r.serve_contact >= 0:
            runs[-1].append(contacts[r.serve_contact])
        elif r.serving_team == "them":
            runs.append([])
    for run in runs if cfg["runs"] else []:
        votes: dict[int, float] = defaultdict(float)
        for c in run:
            if c.jersey in liberos:
                c.jersey = c.player_confidence = None   # not the server
            if c.jersey is not None:
                votes[c.jersey] += c.player_confidence or 0.0
        if not votes:
            continue
        number, w = max(votes.items(), key=lambda kv: kv[1])
        share = w / sum(votes.values())
        if share < cfg["run_min_share"]:
            how["run with two servers (left as is)"] += 1
            continue
        conf = round(share * max(c.player_confidence or 0.0 for c in run if c.jersey == number),
                     3)
        if conf < keep:
            continue
        for c in run:
            if c.jersey != number:
                how["serve named from its run"] += 1
                c.jersey, c.player_confidence = number, conf
    return how
