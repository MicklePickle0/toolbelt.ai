"""Find the stretches of play in a video, so ball detection can skip the time between rallies.

Scores on-court motion every 0.1 s (cached in data/cache/<id>/activity.npz, shared with
sample_active_frames.py) and writes data/cache/<id>/play_spans.json: [first, end) frame
ranges that detect_ball.py processes (detect.ball_play_only). Fast: the video is decoded
small (480 px) on the GPU.

With labelled rallies (data/labels/<id>/events.csv) it checks every labelled rally is
covered, from `check_lead_s` before its start (the serve toss) to its end; any that
isn't is listed. Tune `activity.play:` until none are missed before trusting it on a
new venue.

Usage:
  python scripts/find_play.py --video titan-rise_set1 [--video KEY ...] [--recompute]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from setter_eyes import config
from setter_eyes.ingest import probe
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels
from setter_eyes.rally.activity import activity_series, play_spans


def _clock(frame: int, fps: float) -> str:
    m, s = divmod(frame / fps, 60)
    return f"{int(m)}:{s:04.1f}"


def run(manifest, key: str, recompute: bool) -> None:
    pipeline = config.load("pipeline")
    cfg = pipeline["activity"]
    entry = manifest.video(key)
    cache = config.repo_path(pipeline["paths"]["cache_dir"]) / key
    cache.mkdir(parents=True, exist_ok=True)
    info = probe(entry.path)
    n_frames = info.frame_count or int(info.duration * info.fps)
    act = cache / "activity.npz"
    if act.exists() and not recompute:
        d = np.load(act)
        idx, a = d["idx"], d["activity"]
    else:
        print(f"{key}: scoring motion ...", flush=True)
        idx, a = activity_series(Path(entry.path), cfg)
        np.savez_compressed(act, idx=idx, activity=a)
    spans = play_spans(idx, a, info.fps, n_frames, cfg["play"])
    kept = sum(e - s for s, e in spans) / max(1, n_frames)
    (cache / "play_spans.json").write_text(json.dumps(
        {"video": key, "fps": info.fps, "frame_count": n_frames, "kept_share": round(kept, 4),
         "settings": cfg["play"], "spans": [list(s) for s in spans]}, indent=1),
        encoding="utf-8")
    print(f"{key}: {len(spans)} play spans, {kept:.0%} of the video kept "
          f"-> {cache / 'play_spans.json'}")

    if entry.events is None:
        return
    vl = load_video_labels(manifest, entry)
    if not vl.ok:
        print("  labels have errors (scripts/ingest_labels.py --check); not checked")
        return
    lead = round(cfg["play"]["check_lead_s"] * info.fps)
    missed = []
    for r in vl.rallies:
        lo, hi = max(0, r.start_frame - lead), r.end_frame + 1
        covered = sum(max(0, min(e, hi) - max(s, lo)) for s, e in spans)
        if covered < hi - lo:
            missed.append((r, 1 - covered / (hi - lo)))
    print(f"  labelled rallies fully covered: {len(vl.rallies) - len(missed)}/{len(vl.rallies)}")
    for r, share in missed:
        print(f"  NOT COVERED: set {r.set_number} rally {r.rally_number} "
              f"({_clock(r.start_frame, info.fps)}-{_clock(r.end_frame, info.fps)}): "
              f"{share:.0%} of it outside the play spans")


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", required=True, help="manifest key")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--recompute", action="store_true", help="score the motion again")
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    for key in args.video:
        run(manifest, key, args.recompute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
