"""Pose of our players and their jersey numbers (Milestone 7; roadmap B2-B4).

For every detected sample (detect.person_sample_hz, 10 Hz) of a person on our court at
that moment (track/roles.py ours_now) at least `identity.pose.min_box_height_px` tall:
RTMPose keypoints
(rtmlib, ONNX on the GPU). On back views the back number is read with PARSeq, limited to
the video's roster (identity/numbers.py), and each read votes for its track.

Writes
  data/cache/<id>/pose.npz           keypoints per (frame, track): who touched the ball
                                     (identity/toucher.py, used by find_rallies.py)
  data/cache/<id>/numbers.json       stretches of each track with one number (a summary;
                                     find_rallies.py recomputes them from the reads)
  data/cache/<id>/number_reads.npz   every read (track, frame, log P per roster number),
                                     to tune the voting without the GPU
Needs people_tracks.npz / people_roles.json (scripts/track_people.py) and a roster
assigned to the video (Rosters page) for the numbers; without one only the pose is done.
Settings: `identity:` in config/pipeline.yaml.

Usage:
  python scripts/pose_people.py --video pakmen-black_set2 [--seconds 120]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from setter_eyes import config
from setter_eyes import rosters as rost
from setter_eyes.identity.numbers import NumberReader, back_crop, counted, stretches, torso_lab
from setter_eyes.identity.toucher import Poses
from setter_eyes.ingest import iter_frames, probe
from setter_eyes.labels import load_manifest
from setter_eyes.pose import pose_batch
from setter_eyes.track.people import PeopleTracks
from setter_eyes.track.roles import load_roles, ours_now

PROGRESS_EVERY_S = 20


def roster_numbers(key: str, rosters_dir: Path) -> list[int]:
    slug = rost.load_assignments(rosters_dir).get(key)
    roster = rost.load_all(rosters_dir).get(slug) if slug else None
    return sorted(p.number for p in roster.players) if roster else []


def main() -> int:
    pipeline = config.load("pipeline")
    paths = pipeline["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--seconds", type=float, help="only the first this many seconds (a test)")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    cache = config.repo_path(paths["cache_dir"]) / entry.key
    info = probe(entry.path)
    cfg = config.scale_px(pipeline["identity"], info.width)
    people = PeopleTracks.load(cache / "people_tracks.npz")
    roles = load_roles(cache / "people_roles.json")
    # Ours from where they are at that moment (track/roles.py ours_now, rally or not): a
    # track's lifetime role misses the setter and hitters at the net (tracks swap teams
    # there) and short tracks ("other"). Not the filled-in rows (source 1): those have no
    # detection; corrected positions (2, e.g. mid-jump) keep theirs.
    track_cfg = pipeline["track"]
    ours = np.array([ours_now(xy, roles.get(int(t), {}).get("role"), True, track_cfg)
                     for xy, t in zip(people.court_xy_smooth, people.track_id, strict=True)])
    tall = (people.box[:, 3] - people.box[:, 1]) >= cfg["pose"]["min_box_height_px"]
    sel = np.flatnonzero((people.source != 1) & ours & tall)
    if args.seconds:
        sel = sel[people.frame[sel] < args.seconds * info.fps]
    by_frame: dict[int, np.ndarray] = {}
    order = sel[np.argsort(people.frame[sel], kind="stable")]
    frames, starts = np.unique(people.frame[order], return_index=True)
    for f, rows in zip(frames, np.split(order, starts[1:]), strict=True):
        by_frame[int(f)] = rows
    print(f"{len(sel)} player samples in {len(frames)} frames to pose")

    # GPU libraries only when running. torch first: onnxruntime's CUDA 13 libraries
    # (nvidia-* wheels, loaded by preload_dlls) loaded before torch's CUDA 12.8 ones stop
    # torch from loading; the other way round both work (2026-10-31).
    import onnxruntime
    import torch  # noqa: F401
    from rtmlib import RTMPose
    onnxruntime.preload_dlls()
    p_cfg = cfg["pose"]
    pose = RTMPose(p_cfg["model"], model_input_size=tuple(p_cfg["input_size"]),
                   backend="onnxruntime", device="cuda")
    print("pose runs on: " + ", ".join(pose.session.get_providers()))
    numbers = roster_numbers(entry.key, config.repo_path(paths["rosters_dir"]))
    reader = NumberReader(numbers) if numbers else None
    n_cfg = cfg["numbers"]
    print("roster numbers: " + (", ".join(map(str, numbers)) if numbers else
                                "none (no roster assigned): pose only"))

    out = {k: [] for k in ("frame", "track_id", "box", "kp", "kp_score", "torso_lab")}
    crops, crop_tracks, crop_frames, read_scores = [], [], [], []

    def flush():
        if crops:
            read_scores.append(reader.scores(crops))
            crops.clear()

    waiting: list[tuple] = []                        # (frame index, image, rows) not posed yet

    def pose_waiting():
        items = [(img, people.box[r].astype(float).tolist())
                 for _, img, rows in waiting for r in rows]
        kps, scores = pose_batch(pose, items)
        k = 0
        for index, img, rows in waiting:
            for r in rows:
                kp, sc = kps[k], scores[k]
                k += 1
                out["frame"].append(index)
                out["track_id"].append(int(people.track_id[r]))
                out["box"].append(people.box[r])
                out["kp"].append(kp)
                out["kp_score"].append(sc)
                out["torso_lab"].append(torso_lab(img, kp, sc, n_cfg["keypoint_min_score"]))
                if reader is None:
                    continue
                box = back_crop(kp, sc, n_cfg)
                if box is None or box[3] - box[1] < cfg["min_crop_height_px"]:
                    continue
                c = img[box[1]:box[3], box[0]:box[2]]
                if c.size == 0:
                    continue
                crops.append(c.copy())
                crop_tracks.append(int(people.track_id[r]))
                crop_frames.append(index)
                if len(crops) >= 256:
                    flush()
        waiting.clear()

    t0 = last = time.time()
    keep = set(by_frame)
    for n, fr in enumerate(iter_frames(entry.path, keep=keep.__contains__), start=1):
        if fr.index not in keep:
            continue
        waiting.append((fr.index, fr.image, by_frame[fr.index]))
        if sum(len(w[2]) for w in waiting) >= p_cfg["batch"]:
            pose_waiting()
        if time.time() - last >= PROGRESS_EVERY_S:
            last = time.time()
            print(f"frame {n}/{len(frames)} ({n / (last - t0):.1f} frames/s), "
                  f"{len(crop_tracks)} back views read", flush=True)
        if n >= len(frames):
            break
    pose_waiting()
    if reader is not None:
        flush()
    Poses(np.array(out["frame"], np.int64), np.array(out["track_id"], np.int64),
          np.array(out["box"], np.float32).reshape(-1, 4),
          np.array(out["kp"], np.float32).reshape(-1, 17, 2),
          np.array(out["kp_score"], np.float32).reshape(-1, 17),
          np.array(out["torso_lab"], np.float32).reshape(-1, 3)).save(cache / "pose.npz")
    print(f"posed {len(out['frame'])} player samples in {time.time() - t0:.0f} s "
          f"-> {cache / 'pose.npz'}")
    if reader is None:
        return 0
    scores_all = np.concatenate(read_scores) if read_scores else np.zeros((0, len(numbers)))
    tracks = np.array(crop_tracks, np.int64)
    np.savez_compressed(cache / "number_reads.npz", track_id=tracks,
                        frame=np.array(crop_frames, np.int64), log_p=scores_all,
                        roster=np.array(numbers))
    found = stretches(tracks, np.array(crop_frames, np.int64), scores_all, numbers, n_cfg)
    doc = {"video": entry.key, "roster": numbers,
           "stretches": [s.to_dict() for s in sorted(found, key=lambda s: (s.track_id, s.first))]}
    (cache / "numbers.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    sure = sum(s.confidence >= cfg["number_vote_bind_threshold"] for s in found)
    print(f"back views read: {len(tracks)} ({int(counted(scores_all, n_cfg)[0].sum())} "
          f"counted); {len(found)} stretches of a track with one number ({sure} at "
          f"confidence >= {cfg['number_vote_bind_threshold']}) -> {cache / 'numbers.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
