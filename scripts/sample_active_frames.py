"""Extract labelling frames mostly from live play, using a rough activity score.

Per manifest video: score on-court motion (camera pan removed) every 0.1 s, cached in
data/cache/<id>/activity.npz; pick `activity.bursts` runs of `burst_len` consecutive
frames (`active_share` from active play, the rest from downtime), spread over the video;
extract them exactly into data/cache/<id>/frames/ and list them in
data/cache/<id>/bursts.json (label_batch.py make reads it).

Settings: `activity:` in config/pipeline.yaml.

Usage:
  python scripts/sample_active_frames.py [--video KEY ...] [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from setter_eyes import config
from setter_eyes.ingest import frame_filename, probe, read_frames, write_image
from setter_eyes.labels import load_manifest
from setter_eyes.rally.activity import activity_series, choose_bursts


def run(key: str, path: str) -> str:
    pipeline = config.load("pipeline")
    cfg = pipeline["activity"]
    cache = config.repo_path(pipeline["paths"]["cache_dir"]) / key
    cache.mkdir(parents=True, exist_ok=True)
    act = cache / "activity.npz"
    if act.exists():
        d = np.load(act)
        idx, a = d["idx"], d["activity"]
    else:
        idx, a = activity_series(Path(path), cfg)
        np.savez_compressed(act, idx=idx, activity=a)
    info = probe(path)
    n_frames = info.frame_count or int(info.duration * info.fps)
    # Keep clear of the last second: container frame counts can overshoot slightly.
    bursts = choose_bursts(idx, a, n_frames - round(info.fps), cfg, seed=key)
    wanted = [b["start"] + j for b in bursts for j in range(b["len"])]
    frames_dir = cache / "frames"
    todo = [f for f in wanted if not (frames_dir / frame_filename(key, f)).exists()]
    quality = pipeline["ingest"]["jpeg_quality"]
    for frame in read_frames(path, todo):
        write_image(frames_dir / frame_filename(key, frame.index), frame.image, quality)
    (cache / "bursts.json").write_text(json.dumps({"video": key, "bursts": bursts}, indent=1),
                                       encoding="utf-8")
    kinds = [b["kind"] for b in bursts]
    return (f"{key}: {kinds.count('active')} active + {kinds.count('downtime')} downtime bursts, "
            f"{len(todo)} new frames")


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", help="manifest key (default: all)")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--jobs", type=int, default=4, help="videos in parallel")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    entries = [manifest.video(k) for k in args.video] if args.video else manifest.videos
    with ProcessPoolExecutor(args.jobs) as pool:
        for msg in pool.map(run, [e.key for e in entries], [str(e.path) for e in entries]):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
