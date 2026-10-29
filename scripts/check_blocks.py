"""Check point-ending attacks for a stuff block at full frame rate (GPU, a minute or two).

Run after find_rallies.py (it needs the rallies, actions and point winners): for each rally
that ends on an attack the attacking side lost, the ball is re-detected on every frame
around the attack and its path checked for the second sharp turn of a block
(setter_eyes.contacts.blocks). Results go to data/cache/<id>/block_checks.json (kept and
added to across runs); run find_rallies.py again to add the blocks found and re-judge those
outcomes (it reads the file, no GPU needed).

Usage:
  python scripts/check_blocks.py --video titan-rise_set1 [--oof]
  python scripts/find_rallies.py --video titan-rise_set1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from setter_eyes import config
from setter_eyes.ball.tracker import BallPath
from setter_eyes.contacts.blocks import candidates, full_rate_path, second_turn
from setter_eyes.datasets.tiles import tile_boxes
from setter_eyes.detect.tiled import (
    ball_tiling,
    canonical_names,
    load_detector,
    model_ids,
    predict_crop,
)
from setter_eyes.ingest import probe
from setter_eyes.labels import load_manifest


def main() -> int:
    pipeline = config.load("pipeline")
    paths = pipeline["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--oof", action="store_true",
                    help="check the out-of-fold rallies (rallies_oof.json) instead")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    args = ap.parse_args()

    entry = load_manifest(args.manifest).video(args.video)
    cache = config.repo_path(paths["cache_dir"]) / entry.key
    suffix = "_oof" if args.oof else ""
    rallies = json.loads((cache / f"rallies{suffix}.json").read_text(encoding="utf-8"))
    contacts = json.loads((cache / f"contacts{suffix}.json").read_text(encoding="utf-8"))
    rallies, contacts, fps = rallies["rallies"], contacts["contacts"], float(contacts["fps"])
    info = probe(entry.path)
    cfg = config.for_video(pipeline, info.width)
    s = cfg["contacts"]["stuff_block"]

    out = cache / f"block_checks{suffix}.json"
    checks = json.loads(out.read_text(encoding="utf-8"))["checks"] if out.exists() else []
    done = {c["frame"] for c in checks}
    todo = [contacts[i] for i in candidates(rallies, contacts) if contacts[i]["frame"] not in done]
    print(f"{len(todo)} point-ending attacks lost by the attacker to check "
          f"({len(done)} checked before)")
    if todo:
        det = cfg["detect"]
        model = load_detector(pipeline["train"]["model"],
                              config.repo_path(pipeline["models"]["ball_detector"]),
                              ball_tiling(det, info.width)[1], fp16=True)
        ball = {i: c for i, c in canonical_names(model_ids(model, True)).items() if c == "ball"}
        threshold = det["ball_store_threshold"]
        grid, overlap = ball_tiling(det, info.width)[0], cfg["datasets"]["tile_overlap"]
        _, _, cw, ch = tile_boxes(info.width, info.height, grid, overlap)[0]
        path = BallPath.load(cache / "ball_path.npz")

        def detect_crop(bgr, crop):
            return [(box, score) for _, box, score in
                    predict_crop(model, bgr, crop, threshold, ball)]

        for n, c in enumerate(todo, 1):
            frames, uv = full_rate_path(entry.path, path.frames, path.uv, c["t"], fps,
                                        (info.width, info.height), detect_crop, (cw, ch), s,
                                        pipeline["ingest"].get("hw_decode", False))
            turn, dt = second_turn(frames, uv, c["t"], fps, s)
            checks.append({"frame": c["frame"], "t": c["t"], "fps": fps,
                           "second_turn_px_s": round(turn, 1), "turn_dt_s": round(dt, 4),
                           "ball_found": int(np.isfinite(uv).all(axis=1).sum()),
                           "frames": len(frames)})
            print(f"  {n}/{len(todo)} attack at {c['t']:.2f} s ({c['team']}): second turn "
                  f"{turn:.0f} px/s{' -> block' if turn >= s['min_turn_px_s'] else ''}")
            out.write_text(json.dumps({"video": entry.key, "checks": checks}, indent=1),
                           encoding="utf-8")
    blocks = sum(c["second_turn_px_s"] >= s["min_turn_px_s"] for c in checks)
    print(f"{blocks} stuff blocks among {len(checks)} checked -> {out}")
    print(f"next: python scripts/find_rallies.py --video {entry.key}{' --oof' if args.oof else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
