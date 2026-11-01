"""Who touched it: computed jerseys vs the owner's labelled ones (Milestone 7).

For each labelled touch of ours with a jersey:
  1. at the labelled moment: the player whose hands are at the ball (identity/toucher.py,
     ball from the tracked path) and that track's number then (identity/numbers.py), so
     identity is scored apart from touch detection;
  2. the pipeline's touch within 0.3 s, if found (contacts.json after find_rallies.py, or
     contacts_oof.json with --oof): the jersey it was given.
Prints how many got a number and how many of those are right, also by confidence.
Needs scripts/pose_people.py's pose.npz / number_reads.npz.

Usage:
  python scripts/eval_players.py --video titan-rise_set1_1080p [--oof]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from setter_eyes import config
from setter_eyes.ball.tracker import BallPath
from setter_eyes.identity.numbers import load_numbers
from setter_eyes.identity.toucher import Poses, TouchPicker
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels


def report(name: str, truth: list[int], got: list[tuple[int, float] | None]) -> None:
    n = len(truth)
    given = [(t, g) for t, g in zip(truth, got, strict=True) if g is not None]
    right = sum(t == g[0] for t, g in given)
    line = (f"{name}: {len(given)}/{n} given a number ({len(given) / max(n, 1):.0%}), "
            f"right {right}/{len(given)} ({right / max(len(given), 1):.0%})")
    for c in (0.5, 0.7):
        hi = [(t, g) for t, g in given if g[1] >= c]
        if hi:
            line += (f"; confidence >= {c}: {len(hi)}, right "
                     f"{sum(t == g[0] for t, g in hi) / len(hi):.0%}")
    print(line)


def main() -> int:
    pipeline = config.load("pipeline")
    paths, id_cfg = pipeline["paths"], pipeline["identity"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="labelled manifest key")
    ap.add_argument("--oof", action="store_true", help="score contacts_oof.json")
    args = ap.parse_args()
    manifest = load_manifest(config.repo_path(paths["manifest_path"]))
    vl = load_video_labels(manifest, manifest.video(args.video))
    cache = config.repo_path(paths["cache_dir"]) / args.video
    fps = vl.info.fps
    path = BallPath.load(cache / "ball_path.npz")
    picker = TouchPicker(Poses.load(cache / "pose.npz"), fps, id_cfg["toucher"])
    index = load_numbers(cache / "number_reads.npz", fps, id_cfg["numbers"])
    suffix = "_oof" if args.oof else ""
    contacts = json.loads((cache / f"contacts{suffix}.json").read_text(encoding="utf-8"))
    ours = [c for c in contacts["contacts"] if c["kind"] == "touch" and c["team"] == "us"]

    truth, at_label, at_pipeline = [], [], []
    why = Counter()
    for r in vl.rallies:
        for c in r.contacts:
            if c.team != "us" or c.jersey is None:
                continue
            truth.append(int(c.jersey))
            k = int(np.argmin(np.abs(path.frames - c.frame)))
            uv = path.uv[k] if abs(int(path.frames[k]) - c.frame) <= fps * 0.1 else None
            pick = picker.pick(c.frame, uv)
            got = None
            if pick is None:
                why["no hands at the ball" if uv is not None else "ball not tracked then"] += 1
            else:
                n = index.number_at(pick[0], c.frame)
                if n is None:
                    why["no number on that track then"] += 1
                else:
                    got = (n[0], round(pick[1] * n[1], 3))
                    if n[0] != int(c.jersey):
                        why["wrong number"] += 1
            at_label.append(got)
            m = [x for x in ours if abs(x["t"] - c.frame / fps) <= 0.3]
            j = m[0] if m else None
            at_pipeline.append((j["jersey"], j["player_confidence"])
                               if j and j.get("jersey") is not None else None)
    print(f"{args.video}: {len(truth)} labelled touches of ours with a jersey "
          f"({len(set(truth))} players)")
    report("  at the labelled moment", truth, at_label)
    print(f"    misses: {dict(why)}")
    report(f"  pipeline touches ({'contacts' + suffix}.json)", truth, at_pipeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
