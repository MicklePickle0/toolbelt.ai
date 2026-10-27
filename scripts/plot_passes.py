"""Picture the serve-receive pass scores (Milestone 8 visual check).

For every labelled pass of ours (a `receive` with a pass_score in events.csv), where it was
played (the feet of the player who touched it next: stats/pass_score.py `pass_spot`) is drawn on a
map of our half: dot colour = the owner's score, number = the computed score (ringed in
red when they differ), with the target and the draft grade zones from
config/pass_scale.yaml. Also a few video frames with the spot marked.

Writes data/outputs/passes/passes_map.png and passes_frames.jpg.

Usage:
  python scripts/plot_passes.py --video titan-rise_set1 --video mac-chrome_set1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from setter_eyes import config
from setter_eyes.ball.tracker import BallPath
from setter_eyes.calibration.track import PoseTrack
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels
from setter_eyes.stats.pass_score import grade
from setter_eyes.track.people import PeopleTracks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_passes import load_arcs, toucher_xy  # noqa: E402

PX, MARGIN = 70, 60                                   # map pixels per metre, border
SCORE_BGR = {4: (60, 190, 60), 3: (60, 200, 180), 2: (40, 150, 240), 1: (50, 50, 220),
             0: (40, 20, 120)}


def court_px(x: float, y: float) -> tuple[int, int]:
    """Our half, seen from behind our end line: net at the top, x left -> right."""
    return int(MARGIN + x * PX), int(MARGIN + y * PX)


def draw_map(passes: list[dict], scale: dict) -> np.ndarray:
    w, h = 2 * MARGIN + 9 * PX, 2 * MARGIN + 9 * PX + 150
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(img, court_px(0, 0), court_px(9, 9), (40, 40, 40), 2)
    cv2.line(img, court_px(0, 3), court_px(9, 3), (120, 120, 120), 1)        # attack line
    cv2.line(img, court_px(-0.6, 0), court_px(9.6, 0), (180, 0, 180), 4)       # net
    cv2.putText(img, "NET", (court_px(4.2, 0)[0], MARGIN - 12), 0, 0.6, (180, 0, 180), 2)
    tx, ty = scale["target"]["x"], scale["target"]["y"]
    for g in sorted(scale["grades"], key=lambda g: -g["score"]):
        if g["max_depth_error"] is None or g["max_lateral_error"] is None:
            continue
        a = court_px(tx - g["max_lateral_error"], ty - g["max_depth_error"])
        b = court_px(tx + g["max_lateral_error"], ty + g["max_depth_error"])
        cv2.rectangle(img, a, b, SCORE_BGR[g["score"]], 1, cv2.LINE_AA)
        cv2.putText(img, f"{g['score']} zone", (b[0] - 62, b[1] - 6), 0, 0.45,
                    SCORE_BGR[g["score"]], 1)
    cv2.drawMarker(img, court_px(tx, ty), (0, 0, 0), cv2.MARKER_STAR, 22, 2)
    for p in passes:
        u, v = court_px(*np.clip(p["xy"], [-0.5, -0.5], [9.5, 9.5]))
        cv2.circle(img, (u, v), 13, SCORE_BGR[p["label"]], -1, cv2.LINE_AA)
        if p["computed"] != p["label"]:
            cv2.circle(img, (u, v), 15, (0, 0, 230), 2, cv2.LINE_AA)
        cv2.putText(img, str(p["computed"]), (u - 6, v + 6), 0, 0.55, (255, 255, 255), 2)
    y0 = MARGIN + 9 * PX + 40
    cv2.putText(img, "dot colour = your score, number = computed (red ring: they differ)",
                (MARGIN, y0), 0, 0.5, (0, 0, 0), 1)
    for i, s in enumerate((4, 3, 2, 1, 0)):
        cv2.circle(img, (MARGIN + 10 + i * 110, y0 + 35), 10, SCORE_BGR[s], -1)
        cv2.putText(img, f"yours {s}", (MARGIN + 26 + i * 110, y0 + 41), 0, 0.5, (0, 0, 0), 1)
    exact = np.mean([p["computed"] == p["label"] for p in passes])
    near = np.mean([abs(p["computed"] - p["label"]) <= 1 for p in passes])
    cv2.putText(img, f"{len(passes)} passes: computed = yours {exact:.0%}, within 1 {near:.0%}"
                "  (star: setter target)", (MARGIN, y0 + 85), 0, 0.55, (0, 0, 0), 1)
    return img


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", required=True, help="labelled manifest key")
    args = ap.parse_args()
    manifest = load_manifest(config.repo_path(paths["manifest_path"]))
    scale = config.load("pass_scale")
    passes: list[dict] = []
    for key in args.video:
        vl = load_video_labels(manifest, manifest.video(key))
        fps = vl.info.fps
        cache = config.repo_path(paths["cache_dir"]) / key
        path = BallPath.load(cache / "ball_path.npz")
        people = PeopleTracks.load(cache / "people_tracks.npz")
        arcs, g = load_arcs(cache)
        for r in vl.rallies:
            for i, c in enumerate(r.contacts):
                if c.action != "receive" or c.team != "us" or c.pass_score is None:
                    continue
                nxt = r.contacts[i + 1] if i + 1 < len(r.contacts) else None
                if nxt is None or nxt.team != "us":
                    continue                       # overpass / ace: no setter touch
                xy = toucher_xy(path, people, nxt.frame, arcs, g, fps)
                if xy is not None:
                    passes.append({"video": key, "frame": nxt.frame, "t": nxt.frame / fps,
                                   "xy": xy, "label": c.pass_score,
                                   "computed": grade(xy, scale)})
    out = config.repo_path(paths["output_dir"]) / "passes"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "passes_map.png"), draw_map(passes, scale))

    # A few frames: the best, a middling and the worst pass, as the setter plays them.
    picks = []
    for want in (4, 2, 1):
        cand = [p for p in passes if p["label"] == want]
        if cand:
            picks.append(cand[len(cand) // 2])
    tiles = []
    for p in picks:
        cache = config.repo_path(paths["cache_dir"]) / p["video"]
        cams = PoseTrack.load(config.repo_path(paths["calibration_dir"]) /
                              f"{p['video']}_track.npz")
        src = cache / "proxy.mp4"
        cap = cv2.VideoCapture(str(src if src.exists() else manifest.video(p["video"]).path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"])
        ok, img = cap.read()
        if not ok:
            continue
        s = img.shape[1] / cams.base.width
        cam = cams.camera(p["frame"])
        (spot,), _ = cam.project(np.array([[*p["xy"], 0.0]]))
        (tgt,), _ = cam.project(np.array([[scale["target"]["x"], scale["target"]["y"], 0.0]]))
        cv2.drawMarker(img, tuple((tgt * s).round().astype(int)), (255, 255, 255),
                       cv2.MARKER_STAR, 28, 3)
        cv2.circle(img, tuple((spot * s).round().astype(int)), 18, SCORE_BGR[p["computed"]], 4)
        cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(img, f"{p['video']}  {int(p['t'] // 60)}:{p['t'] % 60:04.1f}   pass "
                    f"computed {p['computed']}, yours {p['label']}   (circle: where the setter "
                    f"plays it, star: target)", (10, 30), 0, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (960, 540)))
    if tiles:
        cv2.imwrite(str(out / "passes_frames.jpg"), np.vstack(tiles))
    print(f"{len(passes)} passes -> {out / 'passes_map.png'}, {out / 'passes_frames.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
