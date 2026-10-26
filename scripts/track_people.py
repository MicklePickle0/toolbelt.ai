"""Track people through a video and place them on the court (Milestone 2, steps 2-3).

Needs data/cache/<id>/person_detections.npz (scripts/detect_people.py) and the camera track
data/calibration/<id>_track.npz (scripts/calibrate.py + track_camera.py). Writes
  data/cache/<id>/people_tracks.npz   one row per tracked detection: frame, track id, box,
                                      score, court x/y (metres, from the feet; raw and
                                      smoothed), feet visible
  data/cache/<id>/people_tracks.json  per track: span, length, median court position and
                                      share of time in our half
  data/cache/<id>/people_roles.json   per track: near_player / far_player / other, with
                                      confidence (provisional near-side filter)
Settings: `track:` in config/pipeline.yaml. Check by eye: render_overlay.py --tracks.

Usage:
  python scripts/track_people.py --video titan-rise_set1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from setter_eyes import config
from setter_eyes.calibration.track import PoseTrack
from setter_eyes.labels import load_manifest
from setter_eyes.track.people import link_tracks, save_summary, track_people
from setter_eyes.track.roles import classify_tracks, near_players_per_frame, save_roles


def main() -> int:
    pipeline = config.load("pipeline")
    paths, cfg = pipeline["paths"], pipeline["track"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    cache = config.repo_path(paths["cache_dir"]) / entry.key
    track_path = config.repo_path(paths["calibration_dir"]) / f"{entry.key}_track.npz"
    if not track_path.exists():
        raise SystemExit(f"no camera track at {track_path}; run scripts/track_camera.py")
    cams = PoseTrack.load(track_path)
    d = np.load(cache / "person_detections.npz")
    cfg = config.scale_px(cfg, cams.base.width)          # pixel settings for this video

    t0 = time.time()
    tracks = track_people(d["frames"], d["det_frame"], d["det_box"], d["det_score"],
                          float(d["fps"]), cams.camera, cams.camera(cams.ref_frame),
                          cams.base.height, cfg)
    tracks, joins = link_tracks(tracks, d["frames"], float(d["fps"]), cfg)
    print(f"joined {joins} broken tracks")
    tracks.save(cache / "people_tracks.npz")
    summary = save_summary(tracks, cfg, cache / "people_tracks.json")

    lengths = np.array([s["samples"] for s in summary])
    long = lengths >= 30 * 10                     # at least ~10 s at 30 samples/s
    ours = [s for s in summary if s["samples"] >= 300 and s["share_in_our_half"] >= 0.8]
    print(f"{len(tracks.frame)} tracked detections ({len(tracks.frame) / len(d['det_frame']):.0%}"
          f" of {len(d['det_frame'])}), {len(summary)} tracks, {int(long.sum())} lasting >= 10 s, "
          f"median length {np.median(lengths) / 30:.1f} s")
    print(f"tracks >= 10 s spending >= 80% of their time in our half (+margin): {len(ours)}")
    src = np.bincount(tracks.source, minlength=3) / len(tracks.source)
    print(f"rows: {src[0]:.0%} detected, {src[1]:.0%} filled gaps (player hidden), "
          f"{src[2]:.0%} corrected positions")
    roles = classify_tracks(tracks, float(d["fps"]), cfg)
    save_roles(roles, cache / "people_roles.json")
    counts = {r: sum(x["role"] == r for x in roles) for r in ("near_player", "far_player",
                                                              "other")}
    _, per_frame = near_players_per_frame(tracks, roles)
    q = np.percentile(per_frame, [10, 50, 90]).astype(int)
    print(f"roles: {counts}; near players per frame: median {q[1]} "
          f"(10th-90th percentile {q[0]}-{q[2]})")
    print(f"feet visible: {tracks.feet_visible.mean():.0%}; "
          f"took {time.time() - t0:.0f} s -> {cache / 'people_tracks.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
