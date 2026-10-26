"""What would searching only near the ball (detect/roi.py) cost in accuracy, and save?

Replays a video's saved full-search ball detections through the near-the-ball planner,
then reruns ball tracking, flight fitting, touches and rallies IN MEMORY (nothing in
data/cache is written) and scores both against the video's labels. No GPU.

Usage:
  python scripts/simulate_roi.py --video titan-rise_set1 [--video mac-chrome_set1]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from setter_eyes import config
from setter_eyes.ball.arcs import fit_flights
from setter_eyes.ball.tracker import Detections, track_ball
from setter_eyes.calibration.track import PoseTrack
from setter_eyes.contacts.evaluate import evaluate
from setter_eyes.contacts.scorer import TouchScorer, scored_contacts
from setter_eyes.detect.roi import simulate
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels
from setter_eyes.rally.live import PlayerMotion
from setter_eyes.rally.segment import segment_from_touches
from setter_eyes.track.people import PeopleTracks
from setter_eyes.track.roles import load_roles


def run(dets, cams, people, roles, scorer, cfg):
    bcfg = cfg["ball"]
    path = track_ball(dets, cams.camera, cams.camera(cams.ref_frame), bcfg)
    times = path.frames / dets.fps
    valid = path.measured & np.isfinite(path.uv).all(axis=1)
    arcs = fit_flights(times, path.uv_ref, path.uv, valid,
                       [cams.camera(int(f)) for f in path.frames], path.size_px, bcfg)
    g = np.array([0.0, 0.0, -float(bcfg["gravity"])])
    contacts = scored_contacts(path, arcs, g, dets.fps, people, roles, cfg, scorer)
    rallies = segment_from_touches(path, arcs, contacts, g, dets.fps, cfg,
                                   PlayerMotion(people, dets.fps))
    return path, [c.to_dict() for c in contacts], [r.to_dict() for r in rallies]


def main() -> int:
    cfg = config.load("pipeline")
    paths = cfg["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", required=True, help="labelled manifest key")
    args = ap.parse_args()
    manifest = load_manifest(config.repo_path(paths["manifest_path"]))
    scorer = TouchScorer.load(config.repo_path(cfg["contacts"]["scorer"]["path"]))
    d = cfg["datasets"]
    grid = tuple(int(v) for v in str(d["tiles"]).split("x"))
    for key in args.video:
        entry = manifest.video(key)
        cache = config.repo_path(paths["cache_dir"]) / key
        cams = PoseTrack.load(config.repo_path(paths["calibration_dir"]) / f"{key}_track.npz")
        people = PeopleTracks.load(cache / "people_tracks.npz")
        roles = load_roles(cache / "people_roles.json")
        labels = load_video_labels(manifest, entry).rallies
        full = Detections.load(cache / "ball_detections.npz")
        roi, share, modes = simulate(full, cams.base.width, cams.base.height, grid,
                                     d["tile_overlap"], cfg["detect"]["roi"])
        split = ", ".join(f"{k} {v:.0%}" for k, v in modes.items())
        print(f"== {key}: near-the-ball search needs {share:.0%} of the detector runs "
              f"(~{1 / share:.1f}x faster ball detection; frames: {split}); keeps "
              f"{len(roi.det_score)}/{len(full.det_score)} detections")
        for name, dets in (("full search", full), ("near the ball", roi)):
            path, contacts, rallies = run(dets, cams, people, roles, scorer, cfg)
            s = evaluate(labels, contacts, rallies, dets.fps, cfg).summary()
            seen = float((path.measured & np.isfinite(path.uv).all(axis=1)).mean())
            print(f"  {name:14s} ball seen {seen:.0%}; touches found {s['touch_recall']:.0%}, "
                  f"real {s['touch_precision']:.0%}, team {s['team_accuracy']:.0%}; rallies "
                  f"{s['rallies_found']}/{s['rallies_labelled']}, false {s['false_rallies']}")
    print(json.dumps(cfg["detect"]["roi"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
