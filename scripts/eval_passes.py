"""Compare computed serve-receive pass scores (0-4) with the owner's labelled ones.

For each labelled pass of ours (a `receive` with a pass_score in events.csv), the pass is
located at the feet of the player who touches it next (stats/pass_score.py `pass_spot`:
by the ball's 3D position when known, else the player whose hands are at the ball) and
graded with config/pass_scale.yaml. Prints how often
the computed score matches (exactly and within 1), a confusion table, and distance
thresholds tuned on the other passes (leave-one-out) as a suggestion: the owner decides
the boundaries (CLAUDE.md).

Usage:
  python scripts/eval_passes.py --video titan-rise_set1 --video mac-chrome_set1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from setter_eyes import config
from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels
from setter_eyes.stats.pass_score import grade, pass_spot
from setter_eyes.track.people import PeopleTracks


def load_arcs(cache: Path) -> tuple[list[Arc], np.ndarray]:
    d = json.loads((cache / "ball_arcs.json").read_text(encoding="utf-8"))
    return [Arc(**a) for a in d["arcs"]], np.array([0.0, 0.0, -float(d["gravity"])])


def ball_xyz(arcs: list[Arc], g: np.ndarray, t: float, min_conf: float = 0.4,
             slack_s: float = 0.1) -> np.ndarray | None:
    """The ball's court position at t from a confident flight around it, or None."""
    for a in arcs:
        if a.confidence >= min_conf and a.t0 - slack_s <= t <= a.t0 + a.duration_s + slack_s:
            return a.position(np.array([min(max(t, a.t0), a.t0 + a.duration_s)]), g)[0]
    return None


def toucher_xy(path: BallPath, people: PeopleTracks, frame: int, arcs: list[Arc],
               g: np.ndarray, fps: float) -> np.ndarray | None:
    """Feet (court x, y) of the player who plays the ball at `frame`."""
    return pass_spot(path, people, frame, ball_xyz(arcs, g, frame / fps))


def suggest(dist: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """Distance limits (4, 3, 2) minimising the mean score error."""
    cands = np.arange(0.5, 6.01, 0.25)
    best = None
    for t4 in cands:
        for t3 in cands[cands > t4]:
            for t2 in cands[cands > t3]:
                p = np.where(dist <= t4, 4, np.where(dist <= t3, 3, np.where(dist <= t2, 2, 1)))
                e = float(np.mean(np.abs(p - scores)))
                if best is None or e < best[0]:
                    best = (e, t4, t3, t2)
    return best[1:]


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", required=True, help="labelled manifest key")
    args = ap.parse_args()
    manifest = load_manifest(config.repo_path(paths["manifest_path"]))
    scale = config.load("pass_scale")
    labelled, computed, dist = [], [], []
    for key in args.video:
        vl = load_video_labels(manifest, manifest.video(key))
        cache = config.repo_path(paths["cache_dir"]) / key
        path = BallPath.load(cache / "ball_path.npz")
        people = PeopleTracks.load(cache / "people_tracks.npz")
        arcs, g = load_arcs(cache)
        for r in vl.rallies:
            for i, c in enumerate(r.contacts):
                if c.action != "receive" or c.team != "us" or c.pass_score is None:
                    continue
                nxt = r.contacts[i + 1] if i + 1 < len(r.contacts) else None
                if nxt is None or nxt.team != "us":
                    labelled.append(c.pass_score)
                    computed.append(scale["overpass_score"] if nxt is not None else None)
                    dist.append(np.nan)
                    continue
                xy = toucher_xy(path, people, nxt.frame, arcs, g, vl.info.fps)
                labelled.append(c.pass_score)
                computed.append(grade(xy, scale) if xy is not None else None)
                dist.append(np.hypot(xy[0] - scale["target"]["x"], xy[1] - scale["target"]["y"])
                            if xy is not None else np.nan)
    ok = [i for i, p in enumerate(computed) if p is not None]
    L = np.array([labelled[i] for i in ok])
    P = np.array([computed[i] for i in ok])
    print(f"labelled passes: {len(labelled)}, scored: {len(ok)}")
    print(f"config/pass_scale.yaml grades: exact {np.mean(P == L):.0%}, "
          f"within 1 {np.mean(np.abs(P - L) <= 1):.0%}")
    conf = Counter(zip(L.tolist(), P.tolist(), strict=True))
    print("labelled -> computed: " + ", ".join(f"{a}->{b}: {n}" for (a, b), n in
                                                sorted(conf.items())))
    D = np.array(dist)
    m = np.isfinite(D)
    Ld, Dd = np.array(labelled)[m], D[m]
    for s in sorted(set(Ld.tolist()), reverse=True):
        print(f"  labelled {s}: distance of the setter from the target, median "
              f"{np.median(Dd[Ld == s]):.1f} m (n={int((Ld == s).sum())})")
    loo = []
    for i in range(len(Dd)):
        keep = np.arange(len(Dd)) != i
        t4, t3, t2 = suggest(Dd[keep], Ld[keep])
        loo.append(4 if Dd[i] <= t4 else 3 if Dd[i] <= t3 else 2 if Dd[i] <= t2 else 1)
    loo = np.array(loo)
    t4, t3, t2 = suggest(Dd, Ld)
    print(f"suggestion, distance only (tested on held-out passes): exact {np.mean(loo == Ld):.0%}, "
          f"within 1 {np.mean(np.abs(loo - Ld) <= 1):.0%}; 4 up to {t4} m, 3 up to {t3} m, "
          f"2 up to {t2} m. Few passes: confirm before changing the config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
