"""Find ball contacts, serves and rallies in a video (Milestone 4).

Needs the ball path and arcs (scripts/track_ball.py) and the people tracks and roles
(scripts/track_people.py). Writes
  data/cache/<id>/contacts.json   touches and floor contacts: time, team, toucher's track,
                                  ball position, evidence, confidence, serve flag
  data/cache/<id>/rallies.json    rallies: start (serve) and end (floor / ball lost),
                                  serving team, contacts, net crossings, confidence
Rallies carry no point winner: that comes from the review UI (CLAUDE.md).
Also writes rallies and touches (with actions) to the event DB (paths.db_path), unreviewed,
replacing what was there for this video; --no-db skips it.
Touches are chosen by the learned scorer when its model exists (contacts.scorer.path,
scripts/train_touch_scorer.py), else by the hand-tuned fusion in contacts/events.py.
Settings: `contacts:`, `serve:` and `rally:` in config/pipeline.yaml.
Check by eye: render_overlay.py --rallies.

--oof builds rallies from data/cache/<id>/contacts_oof.json instead (touches chosen
without training on them, from train_touch_scorer.py) and writes rallies_oof.json, for
an honest test of the rally logic with eval_contacts.py --oof.

Usage:
  python scripts/find_rallies.py --video titan-rise_set1 [--oof]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from setter_eyes import config, db
from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath
from setter_eyes.classify.rules import classify_rally, launch_up
from setter_eyes.contacts.events import Contact, find_contacts
from setter_eyes.contacts.scorer import TouchScorer, scored_contacts
from setter_eyes.db.pipeline import write_video
from setter_eyes.labels import load_manifest
from setter_eyes.rally.live import PlayerMotion
from setter_eyes.rally.segment import segment_from_touches, segment_rallies
from setter_eyes.track.people import PeopleTracks
from setter_eyes.track.roles import load_roles


def main() -> int:
    cfg = config.load("pipeline")
    paths = cfg["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--no-db", action="store_true", help="don't write the event DB")
    ap.add_argument("--oof", action="store_true",
                    help="rallies from contacts_oof.json (honest test of the rally logic)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    entry = manifest.video(args.video)
    cache = config.repo_path(paths["cache_dir"]) / entry.key
    path = BallPath.load(cache / "ball_path.npz")
    arcs_doc = json.loads((cache / "ball_arcs.json").read_text(encoding="utf-8"))
    arcs = [Arc(**a) for a in arcs_doc["arcs"]]
    fps = float(arcs_doc["fps"])
    g = np.array([0.0, 0.0, -float(arcs_doc["gravity"])])
    people = PeopleTracks.load(cache / "people_tracks.npz")
    roles = load_roles(cache / "people_roles.json")

    model_path = config.repo_path(cfg["contacts"]["scorer"]["path"])
    suffix = "_oof" if args.oof else ""
    if args.oof:
        doc = json.loads((cache / "contacts_oof.json").read_text(encoding="utf-8"))
        contacts = [Contact.from_dict(c) for c in doc["contacts"]]
        print("touches: out-of-fold (contacts_oof.json)")
    elif model_path.exists():
        scorer = TouchScorer.load(model_path)
        print(f"touches: learned scorer (trained on {', '.join(scorer.trained_on)})")
        contacts = scored_contacts(path, arcs, g, fps, people, roles, cfg, scorer)
    else:
        print("touches: hand-tuned fusion (no learned scorer yet)")
        contacts = find_contacts(path, arcs, g, fps, people, roles, cfg)
    learned = args.oof or model_path.exists()
    if learned:
        rallies = segment_from_touches(path, arcs, contacts, g, fps, cfg,
                                       PlayerMotion(people, fps))
    else:
        rallies = segment_rallies(path, arcs, contacts, g, fps, cfg)
    for r in rallies:                                  # actions, once the rally is known
        touches = [contacts[i] for i in r.contacts if contacts[i].kind == "touch"]
        serve = contacts[r.serve_contact] if r.serve_contact is not None and \
            r.serve_contact >= 0 else None
        launches = [launch_up(c, arcs, cfg["classify"]) for c in touches]
        for c, (action, conf) in zip(touches, classify_rally(touches, serve, r.serving_team,
                                                             cfg["classify"], launches),
                                     strict=True):
            c.action, c.action_confidence = action, conf
    (cache / f"contacts{suffix}.json").write_text(json.dumps(
        {"video": entry.key, "fps": fps, "contacts": [c.to_dict() for c in contacts]},
        indent=1), encoding="utf-8")
    (cache / f"rallies{suffix}.json").write_text(json.dumps(
        {"video": entry.key, "fps": fps, "rallies": [r.to_dict() for r in rallies]},
        indent=1), encoding="utf-8")

    kinds = Counter(c.kind for c in contacts)
    in_rally = sum(len(r.contacts) for r in rallies)
    serve_team = Counter(r.serving_team for r in rallies)
    lengths = np.array([r.end_t - r.start_t for r in rallies]) if rallies else np.zeros(1)
    review = cfg["review"]["default_min_confidence"]
    print(f"contacts: {dict(kinds)} ({sum(c.serve for c in contacts)} look like serves), "
          f"{in_rally} inside rallies")
    traced = sum(r.serve_contact == -2 for r in rallies)
    inferred = sum(r.serve_contact == -3 for r in rallies)
    print(f"rallies: {len(rallies)}; serves found for "
          f"{sum(r.serve_contact is not None for r in rallies)} ({traced} traced back from "
          f"flight, {inferred} inferred from the rally)"
          f" (us {serve_team.get('us', 0)}, them {serve_team.get('them', 0)}); "
          f"ended on the floor: {sum(r.end_reason == 'floor' for r in rallies)}")
    print(f"ended by the live-ball check (touches after the whistle dropped): "
          f"{sum(r.end_reason == 'dead_ball' for r in rallies)}")
    print(f"rally length s: median {np.median(lengths):.1f}, max {lengths.max():.1f}; "
          f"below review confidence {review}: {sum(r.confidence < review for r in rallies)}")
    print(f"-> {cache / f'contacts{suffix}.json'}, {cache / f'rallies{suffix}.json'}")
    if not args.no_db and not args.oof:
        conn = db.connect(config.repo_path(paths["db_path"]))
        w = write_video(conn, manifest, entry, fps, [r.to_dict() for r in rallies],
                        [c.to_dict() for c in contacts])
        conn.close()
        print(f"event DB: {w.rallies} rallies, {w.contacts} touches -> {paths['db_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
