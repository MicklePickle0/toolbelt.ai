"""Train the learned touch scorer on labelled videos (events.csv) and report how it does.

Needs, per video: the pipeline caches (track_ball.py, track_people.py) and
data/labels/<id>/events.csv. Only candidates inside labelled rallies (+1 s) are used.
Reports a validation grouped by rally (each rally scored by a model that never saw it),
then trains on everything and saves it to contacts.scorer.path (config/pipeline.yaml).
find_rallies.py uses the saved model from then on.

The validation is within the training videos. The honest test is a labelled video it
was NOT trained on: scripts/eval_contacts.py warns when you score a training video.

Usage:
  python scripts/train_touch_scorer.py --video titan-rise_set1 [--video mac-chrome_set1]
  python scripts/train_touch_scorer.py --video titan-rise_set1 --no-save   # report only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from setter_eyes import config
from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath
from setter_eyes.contacts.evaluate import evaluate
from setter_eyes.contacts.features import NAMES, candidate_features
from setter_eyes.contacts.scorer import TouchScorer, decode, raw_candidates, touch_targets
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels
from setter_eyes.track.people import PeopleTracks
from setter_eyes.track.roles import load_roles


def video_dataset(entry, manifest, cfg: dict, paths: dict) -> dict:
    vl = load_video_labels(manifest, entry)
    if not vl.ok:
        raise SystemExit(f"{entry.key}: label errors; run ingest_labels.py --check")
    d = config.repo_path(paths["cache_dir"]) / entry.key
    path = BallPath.load(d / "ball_path.npz")
    ad = json.loads((d / "ball_arcs.json").read_text(encoding="utf-8"))
    fps = float(ad["fps"])
    g = np.array([0.0, 0.0, -float(ad["gravity"])])
    people = PeopleTracks.load(d / "people_tracks.npz")
    roles = load_roles(d / "people_roles.json")
    cands = [c for c in raw_candidates(path, [Arc(**a) for a in ad["arcs"]], g, fps, people,
                                       roles, cfg) if c.kind == "touch"]
    X = candidate_features(cands, path, people, fps, cfg["contacts"]["overhead_reach_bodies"])
    t = np.array([c.t for c in cands])
    windows = [(r.start_frame / fps - 1, r.end_frame / fps + 1) for r in vl.rallies]
    rally = np.array([next((j for j, (a, b) in enumerate(windows) if a <= x <= b), -1)
                      for x in t])
    labels = [c for r in vl.rallies for c in r.contacts]
    y, which = touch_targets(t, np.array([c.frame / fps for c in labels]),
                             cfg["contacts"]["scorer"]["match_tolerance_s"])
    ours = np.array([int(labels[w].team == "us") if w >= 0 else -1 for w in which])
    keep = rally >= 0
    return {"key": entry.key, "fps": fps, "labels": vl.rallies, "cands": np.array(cands)[keep],
            "X": X[keep], "t": t[keep], "y": y[keep], "ours": ours[keep], "rally": rally[keep]}


def main() -> int:
    cfg = config.load("pipeline")
    paths = cfg["paths"]
    s = cfg["contacts"]["scorer"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", required=True, help="labelled manifest key")
    ap.add_argument("--no-save", action="store_true", help="validate only")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    data = [video_dataset(manifest.video(k), manifest, cfg, paths) for k in args.video]
    X = np.vstack([d["X"] for d in data])
    y = np.concatenate([d["y"] for d in data])
    ours = np.concatenate([d["ours"] for d in data])
    groups = np.concatenate([d["rally"] + 10_000 * i for i, d in enumerate(data)])
    print(f"{len(y)} candidates, {y.sum()} touches, from {len(set(groups))} rallies in "
          f"{', '.join(args.video)}")

    # Validation grouped by rally.
    p_touch, p_ours = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(min(s["cv_folds"], len(set(groups)))).split(X, y, groups):
        m = TouchScorer.fit(X[tr], y[tr], ours[tr], [], s)
        p_touch[te], p_ours[te] = m.predict(X[te])
    off = 0
    for d in data:
        n = len(d["y"])
        pt, po = p_touch[off:off + n], p_ours[off:off + n]
        off += n
        kept = []
        for j in decode(d["t"], pt, s["threshold"], s["nms_s"]):
            c = d["cands"][j].to_dict()
            c["confidence"], c["team"] = float(pt[j]), "us" if po[j] >= 0.5 else "them"
            kept.append(c)
        r = evaluate(d["labels"], kept, [], d["fps"], cfg).summary()
        print(f"  {d['key']} (validation by rally): touches found {r['touch_recall']:.0%}, "
              f"of predicted real {r['touch_precision']:.0%}, team right "
              f"{r['team_accuracy']:.0%}, timing {r['median_time_error_s']:.2f} s")

    if args.no_save:
        return 0
    model = TouchScorer.fit(X, y, ours, list(args.video), s)
    imp = np.abs(np.corrcoef(np.nan_to_num(X).T, y)[-1, :-1])
    top = ", ".join(NAMES[i] for i in np.argsort(-np.nan_to_num(imp))[:6])
    out = model.save(config.repo_path(s["path"]))
    print(f"saved {out} (trained on {', '.join(args.video)})\n"
          f"most correlated with real touches: {top}\n"
          f"next: python scripts/find_rallies.py --video <id>  (uses the model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
