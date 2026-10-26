"""Train the learned touch scorer on labelled videos (events.csv) and report how it does.

Needs, per video: the pipeline caches (track_ball.py, track_people.py) and
data/labels/<id>/events.csv. Only candidates inside labelled rallies (+1 s) are used.
Reports a validation grouped by rally (each rally scored by a model that never saw it),
then trains on everything and saves it to contacts.scorer.path (config/pipeline.yaml).
find_rallies.py uses the saved model from then on.

Also writes data/cache/<id>/contacts_oof.json: touches as the pipeline would choose them
WITHOUT having trained on them (labelled rallies: out-of-fold; everything else: the final
model, which never saw it), plus floor contacts. find_rallies.py --oof and
eval_contacts.py --oof test the rally logic on those, so it isn't flattered either.

The validation is within the training videos. The honest test is a labelled video it
was NOT trained on: scripts/eval_contacts.py warns when you score a training video.

--reviewed adds rallies reviewed in the review UI (event DB, paths.db_path) as labels:
those with every touch confirmed, and those dismissed as not a rally (their pipeline
touches are false ones). Where events.csv also covers a rally, events.csv wins. A video
with reviews but no events.csv can be trained on this way.

Usage:
  python scripts/train_touch_scorer.py --video titan-rise_set1 [--video mac-chrome_set1]
  python scripts/train_touch_scorer.py --video titan-rise_set1 --no-save   # report only
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from setter_eyes import config, db
from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath
from setter_eyes.contacts.evaluate import evaluate
from setter_eyes.contacts.events import _merge
from setter_eyes.contacts.features import NAMES, candidate_features
from setter_eyes.contacts.scorer import TouchScorer, decode, raw_candidates, touch_targets
from setter_eyes.db.read import reviewed_for_training
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels
from setter_eyes.track.people import PeopleTracks
from setter_eyes.track.roles import load_roles


def training_rallies(entry, manifest, reviewed_db) -> list:
    """events.csv rallies, plus reviewed rallies (if `reviewed_db`) not already labelled."""
    rallies = []
    if entry.events is not None:
        vl = load_video_labels(manifest, entry)
        if not vl.ok:
            raise SystemExit(f"{entry.key}: label errors; run ingest_labels.py --check")
        rallies = list(vl.rallies)
    if reviewed_db is not None:
        extra = [r for r in reviewed_for_training(reviewed_db, entry.key)
                 if not any(min(r.end_frame, x.end_frame) > max(r.start_frame, x.start_frame)
                            for x in rallies)]
        print(f"  {entry.key}: {len(rallies)} labelled rallies + {len(extra)} reviewed")
        rallies += extra
    if not rallies:
        raise SystemExit(f"{entry.key}: no labels (events.csv) and no reviewed rallies")
    return sorted(rallies, key=lambda r: r.start_frame)


def video_dataset(entry, manifest, cfg: dict, paths: dict, reviewed_db=None) -> dict:
    rallies = training_rallies(entry, manifest, reviewed_db)
    d = config.repo_path(paths["cache_dir"]) / entry.key
    path = BallPath.load(d / "ball_path.npz")
    ad = json.loads((d / "ball_arcs.json").read_text(encoding="utf-8"))
    fps = float(ad["fps"])
    cfg = config.for_video(cfg, int(ad.get("width", 3840)))   # pixel settings for this video
    g = np.array([0.0, 0.0, -float(ad["gravity"])])
    people = PeopleTracks.load(d / "people_tracks.npz")
    roles = load_roles(d / "people_roles.json")
    every = raw_candidates(path, [Arc(**a) for a in ad["arcs"]], g, fps, people, roles, cfg)
    c = cfg["contacts"]
    floors = _merge([x for x in every if x.kind == "floor"
                     and x.confidence >= c["min_confidence"]], c["merge_window_s"])
    cands = [x for x in every if x.kind == "touch"]
    X = candidate_features(cands, path, people, fps, cfg["contacts"]["overhead_reach_bodies"])
    t = np.array([c.t for c in cands])
    windows = [(r.start_frame / fps - 1, r.end_frame / fps + 1) for r in rallies]
    rally = np.array([next((j for j, (a, b) in enumerate(windows) if a <= x <= b), -1)
                      for x in t])
    labels = [c for r in rallies for c in r.contacts]
    y, which = touch_targets(t, np.array([c.frame / fps for c in labels]),
                             cfg["contacts"]["scorer"]["match_tolerance_s"])
    ours = np.array([int(labels[w].team == "us") if w >= 0 else -1 for w in which])
    keep = rally >= 0
    return {"key": entry.key, "fps": fps, "labels": rallies, "cands": np.array(cands)[keep],
            "X": X[keep], "t": t[keep], "y": y[keep], "ours": ours[keep], "rally": rally[keep],
            "outside": {"cands": np.array(cands)[~keep], "X": X[~keep], "t": t[~keep]},
            "floors": floors, "cache": d}


def _chosen(cands, t, p_touch, p_ours, s: dict) -> list:
    """The candidates decode() keeps, with the scorer's confidence and team."""
    out = []
    for j in decode(t, p_touch, s["threshold"], s["nms_s"]):
        c = copy.copy(cands[j])
        c.confidence = round(float(p_touch[j]), 3)
        c.team = "us" if p_ours[j] >= 0.5 else "them"
        out.append(c)
    return out


def main() -> int:
    cfg = config.load("pipeline")
    paths = cfg["paths"]
    s = cfg["contacts"]["scorer"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", required=True, help="labelled manifest key")
    ap.add_argument("--no-save", action="store_true", help="validate only")
    ap.add_argument("--reviewed", action="store_true",
                    help="also learn from rallies reviewed in the review UI")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    reviewed_db = db.connect(config.repo_path(paths["db_path"])) if args.reviewed else None
    data = [video_dataset(manifest.video(k), manifest, cfg, paths, reviewed_db)
            for k in args.video]
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
        d["oof"] = (pt, po)
        kept = [c.to_dict() for c in _chosen(d["cands"], d["t"], pt, po, s)]
        r = evaluate(d["labels"], kept, [], d["fps"], cfg).summary()
        print(f"  {d['key']} (validation by rally): touches found {r['touch_recall']:.0%}, "
              f"of predicted real {r['touch_precision']:.0%}, team right "
              f"{r['team_accuracy']:.0%}, timing {r['median_time_error_s']:.2f} s")

    model = TouchScorer.fit(X, y, ours, list(args.video), s)
    for d in data:                     # everything predicted without training on it
        o = d["outside"]
        po_t, po_o = model.predict(o["X"])
        t_all = np.concatenate([d["t"], o["t"]])
        order = np.argsort(t_all, kind="stable")
        cands = np.concatenate([d["cands"], o["cands"]])[order]
        pt = np.concatenate([d["oof"][0], po_t])[order]
        po = np.concatenate([d["oof"][1], po_o])[order]
        chosen = _chosen(cands, t_all[order], pt, po, s)
        contacts = sorted(chosen + d["floors"], key=lambda c: c.t)
        (d["cache"] / "contacts_oof.json").write_text(json.dumps(
            {"video": d["key"], "fps": d["fps"], "out_of_fold": True,
             "contacts": [c.to_dict() for c in contacts]}, indent=1), encoding="utf-8")
        print(f"  {d['key']}: out-of-fold touches -> {d['cache'] / 'contacts_oof.json'}")
    if args.no_save:
        return 0
    imp = np.abs(np.corrcoef(np.nan_to_num(X).T, y)[-1, :-1])
    top = ", ".join(NAMES[i] for i in np.argsort(-np.nan_to_num(imp))[:6])
    out = model.save(config.repo_path(s["path"]))
    print(f"saved {out} (trained on {', '.join(args.video)})\n"
          f"most correlated with real touches: {top}\n"
          f"next: python scripts/find_rallies.py --video <id>  (uses the model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
