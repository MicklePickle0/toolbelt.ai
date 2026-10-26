"""Compare computed serve-receive pass scores (0-4) with the owner's labelled ones.

For each labelled pass of ours (a `receive` with a pass_score in events.csv), the pass is
located at the feet of the player who touches it next (stats/pass_score.py: the person
nearest the ball at that touch) and graded with config/pass_scale.yaml. Prints how often
the computed score matches (exactly and within 1), a confusion table, and distance
thresholds tuned on the other passes (leave-one-out) as a suggestion: the owner decides
the boundaries (CLAUDE.md).

Usage:
  python scripts/eval_passes.py --video titan-rise_set1 --video mac-chrome_set1
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from setter_eyes import config
from setter_eyes.ball.tracker import BallPath
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels
from setter_eyes.stats.pass_score import grade
from setter_eyes.track.people import PeopleTracks


def toucher_xy(path: BallPath, people: PeopleTracks, frame: int) -> np.ndarray | None:
    """Feet (court x, y) of the person on our side nearest the ball at `frame`."""
    k = int(np.argmin(np.abs(path.frames - frame)))
    uv = path.uv[k]
    sel = np.flatnonzero(np.abs(people.frame - frame) <= 2)
    if not np.isfinite(uv).all() or not len(sel):
        return None
    best, best_d = None, np.inf
    for j in sel:
        x1, y1, x2, y2 = people.box[j]
        h = max(1.0, y2 - y1)
        d = np.hypot(max(x1 - uv[0], 0.0, uv[0] - x2),
                     max(y1 - 0.8 * h - uv[1], 0.0, uv[1] - y2)) / h   # reach above the head
        xy = people.court_xy_smooth[j]
        if d < best_d and np.isfinite(xy).all() and xy[1] > -0.5:
            best, best_d = xy, d
    return best if best_d < 0.5 else None


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
                xy = toucher_xy(path, people, nxt.frame)
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
