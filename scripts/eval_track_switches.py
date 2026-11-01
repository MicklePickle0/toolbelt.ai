"""How much the tracker's id switches between players would cost jerseys, and how much the
local binding (identity/numbers.py NumberIndex) saves.

For each labelled touch of ours with a jersey, the toucher's track (identity/toucher.py) is
named two ways: one number for the whole track (its most read number over its life, as if
a track were one player), and the number its reads show around the touch (NumberIndex, what
the pipeline uses). Where the two differ, the track passed between players; this counts
which one matches the owner's label. Also: how many tracks carry 2+ players' numbers.

Usage:
  python scripts/eval_track_switches.py --video titan-rise_set1_1080p
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from setter_eyes import config
from setter_eyes.ball.tracker import BallPath
from setter_eyes.identity.numbers import counted_reads, load_numbers
from setter_eyes.identity.toucher import Poses, TouchPicker
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels


def main() -> int:
    pipeline = config.load("pipeline")
    paths, id_cfg = pipeline["paths"], pipeline["identity"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="labelled manifest key")
    args = ap.parse_args()
    manifest = load_manifest(config.repo_path(paths["manifest_path"]))
    vl = load_video_labels(manifest, manifest.video(args.video))
    cache = config.repo_path(paths["cache_dir"]) / args.video
    fps = vl.info.fps
    path = BallPath.load(cache / "ball_path.npz")
    picker = TouchPicker(Poses.load(cache / "pose.npz"), fps, id_cfg["toucher"])
    reads = cache / "number_reads.npz"
    index = load_numbers(reads, fps, id_cfg["numbers"])
    tracks, _, numbers = counted_reads(reads, id_cfg["numbers"])
    whole = {}
    multi = 0
    for t in np.unique(tracks):
        c = Counter(numbers[tracks == t].tolist())
        whole[int(t)] = c.most_common(1)[0][0]
        # 2+ numbers each with 5+ reads: the track carried 2+ players
        multi += sum(n >= 5 for n in c.values()) >= 2
    print(f"{args.video}: {len(whole)} tracks with counted reads, {multi} of them read as 2+ "
          f"players (5+ reads each): the tracker passed the id on")

    res = Counter()
    seen = []                                  # (track, bound number or None, true jersey)
    for r in vl.rallies:
        for c in r.contacts:
            if c.team != "us" or c.jersey is None:
                continue
            k = int(np.argmin(np.abs(path.frames - c.frame)))
            pick = picker.pick(c.frame, path.uv[k])
            if pick is None or pick[0] not in whole:
                continue
            local = index.number_at(pick[0], c.frame)
            w, truth = whole[pick[0]], int(c.jersey)
            seen.append((pick[0], local[0] if local else None, truth))
            if local is None:
                res["no local reads"] += 1
                res["whole-track right (no local reads)"] += w == truth
                continue
            res["touches"] += 1
            if local[0] == w:
                res["same number"] += 1
                res["same, right"] += w == truth
            else:
                res["differ (the track switched players)"] += 1
                res["differ: local right"] += local[0] == truth
                res["differ: whole-track right"] += w == truth
    for k, v in res.items():
        print(f"  {k}: {v}")
    # Switches as pairs of labelled touches: one identity, two different real players.
    # Tracker: same track id. Bound: same track id AND same number read there (the track's
    # stretch for that player); a switch the numbers caught splits the identity.
    raw = bound = raw_n = bound_n = 0
    for i, (ta, na, ja) in enumerate(seen):
        for tb, nb, jb in seen[i + 1:]:
            if ta != tb:
                continue
            raw_n += 1
            raw += ja != jb
            if na is not None and na == nb:
                bound_n += 1
                bound += ja != jb
    print(f"  identity switches, pairs of labelled touches on one identity with two real "
          f"players: tracker {raw}/{raw_n} ({raw / max(raw_n, 1):.0%}), after binding numbers "
          f"{bound}/{bound_n} ({bound / max(bound_n, 1):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
