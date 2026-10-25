"""Find ball contacts, serves and rallies in a video (Milestone 4).

Needs the ball path and arcs (scripts/track_ball.py) and the people tracks and roles
(scripts/track_people.py). Writes
  data/cache/<id>/contacts.json   touches and floor contacts: time, team, toucher's track,
                                  ball position, evidence, confidence, serve flag
  data/cache/<id>/rallies.json    rallies: start (serve) and end (floor / ball lost),
                                  serving team, contacts, net crossings, confidence
Rallies carry no point winner: that comes from the review UI (CLAUDE.md).
Touches are chosen by the learned scorer when its model exists (contacts.scorer.path,
scripts/train_touch_scorer.py), else by the hand-tuned fusion in contacts/events.py.
Settings: `contacts:`, `serve:` and `rally:` in config/pipeline.yaml.
Check by eye: render_overlay.py --rallies.

Usage:
  python scripts/find_rallies.py --video titan-rise_set1
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
from setter_eyes.contacts.events import find_contacts
from setter_eyes.contacts.scorer import TouchScorer, scored_contacts
from setter_eyes.labels import load_manifest
from setter_eyes.rally.segment import segment_rallies
from setter_eyes.track.people import PeopleTracks
from setter_eyes.track.roles import load_roles


def main() -> int:
    cfg = config.load("pipeline")
    paths = cfg["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    cache = config.repo_path(paths["cache_dir"]) / entry.key
    path = BallPath.load(cache / "ball_path.npz")
    arcs_doc = json.loads((cache / "ball_arcs.json").read_text(encoding="utf-8"))
    arcs = [Arc(**a) for a in arcs_doc["arcs"]]
    fps = float(arcs_doc["fps"])
    g = np.array([0.0, 0.0, -float(arcs_doc["gravity"])])
    people = PeopleTracks.load(cache / "people_tracks.npz")
    roles = load_roles(cache / "people_roles.json")

    model_path = config.repo_path(cfg["contacts"]["scorer"]["path"])
    if model_path.exists():
        scorer = TouchScorer.load(model_path)
        print(f"touches: learned scorer (trained on {', '.join(scorer.trained_on)})")
        contacts = scored_contacts(path, arcs, g, fps, people, roles, cfg, scorer)
    else:
        print("touches: hand-tuned fusion (no learned scorer yet)")
        contacts = find_contacts(path, arcs, g, fps, people, roles, cfg)
    rallies = segment_rallies(path, arcs, contacts, g, fps, cfg)
    (cache / "contacts.json").write_text(json.dumps(
        {"video": entry.key, "fps": fps, "contacts": [c.to_dict() for c in contacts]},
        indent=1), encoding="utf-8")
    (cache / "rallies.json").write_text(json.dumps(
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
    print(f"rallies: {len(rallies)}; serves found for "
          f"{sum(r.serve_contact is not None for r in rallies)} ({traced} traced back from flight)"
          f" (us {serve_team.get('us', 0)}, them {serve_team.get('them', 0)}); "
          f"ended on the floor: {sum(r.end_reason == 'floor' for r in rallies)}")
    print(f"rally length s: median {np.median(lengths):.1f}, max {lengths.max():.1f}; "
          f"below review confidence {review}: {sum(r.confidence < review for r in rallies)}")
    print(f"-> {cache / 'contacts.json'}, {cache / 'rallies.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
