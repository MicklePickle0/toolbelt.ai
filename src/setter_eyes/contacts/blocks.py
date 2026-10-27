"""Stuff blocks: the big blocks that win a point.

Owner (2026-10-27): the blocks worth counting are the ones that win points; the many
fingertip touches don't matter (an attacker who swung and whose side scored has a kill,
touched or not). A block comes 0.03-0.23 s after the attack (labelled), below what the
30 Hz ball path resolves, so the attack and the block look like one touch.

So only where it changes a stat: a rally that ends on an attack the attacking side lost
(point winner from rally/winner.py). There, the ball is re-detected at the video's full
frame rate around the attack (scripts/check_blocks.py, GPU: one crop per frame where the
30 Hz path puts the ball) and its image path checked for a second sharp turn
`min_gap_s`-`max_gap_s` after the attack's own one, a peak of its own (`valley_ratio` x the
lowest turn between them). Strong enough (`min_turn_px_s`, 4K px/s) means the ball came
back off a block: a block touch by the other side is added and the outcome becomes
block_point (the other side's), else it stays an attack error (net, out).
Labelled point-ending attacks the attacker lost: both stuff blocks had a second turn of
>= 2169 px/s, the 19 attack errors (and others) <= 1565. Only 2 examples, so added blocks
carry `confidence` and go to review.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace

import numpy as np

from setter_eyes.classify.outcome import point_outcome
from setter_eyes.contacts.events import Contact


def second_turn(frames: np.ndarray, uv: np.ndarray, t_touch: float, fps: float,
                cfg: dict) -> tuple[float, float]:
    """(strength px/s, seconds after the touch) of the strongest second turn after the
    touch's own turn in a full-rate image path; (0, 0) if none."""
    lag = cfg["lag_samples"]
    t = np.asarray(frames, float) / fps
    uv = np.asarray(uv, float)
    ok = np.isfinite(uv).all(axis=1)
    d = np.full(len(t), np.nan)
    for j in range(lag, len(t) - lag):
        if ok[j] and ok[j - lag] and ok[j + lag]:
            v_in = (uv[j] - uv[j - lag]) / (t[j] - t[j - lag])
            v_out = (uv[j + lag] - uv[j]) / (t[j + lag] - t[j])
            d[j] = np.linalg.norm(v_out - v_in)
    near = np.flatnonzero((np.abs(t - t_touch) <= cfg["own_turn_s"]) & np.isfinite(d))
    if not len(near):
        return 0.0, 0.0
    # The attack's own turn: the turn peak nearest the touch (not the strongest: a block
    # 0.05 s later can turn the ball harder than the attack did).
    dz = np.nan_to_num(d)
    peaks = [j for j in near if dz[j] >= dz[max(0, j - 1)] and dz[j] >= dz[min(len(d) - 1, j + 1)]]
    own = min(peaks, key=lambda j: abs(t[j] - t_touch)) if peaks else near[np.argmax(d[near])]
    best, at = 0.0, 0.0
    after = (t >= t[own] + cfg["min_gap_s"]) & (t <= t[own] + cfg["max_gap_s"]) & np.isfinite(d)
    for j in np.flatnonzero(after):
        between = d[own + 1:j]
        if not np.isfinite(between).any():
            continue
        if d[j] > cfg["valley_ratio"] * np.nanmin(between) and d[j] > best:
            best, at = float(d[j]), float(t[j] - t_touch)
    return best, at


def candidates(rallies: list[dict], contacts: list[dict]) -> list[int]:
    """Contact indices of point-ending attacks whose side lost the point."""
    out = []
    for r in rallies:
        touches = [i for i in r["contacts"] if contacts[i]["kind"] == "touch"]
        if not touches or not r.get("point_winner"):
            continue
        last = contacts[touches[-1]]
        if last.get("action") == "attack" and last["team"] and last["team"] != r["point_winner"]:
            out.append(touches[-1])
    return out


def full_rate_path(video, path_frames: np.ndarray, path_uv: np.ndarray, t: float, fps: float,
                   size: tuple[int, int], detect_crop, crop_wh: tuple[int, int], cfg: dict,
                   hw_decode: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The ball on every frame of [t - window_before_s, t + window_after_s]: decoded with
    ffmpeg (seeking), detected in one crop around where the 30 Hz path puts it.
    detect_crop(bgr, (x, y, w, h)) -> [(box, score)]. -> (frames, uv) with NaN where missed."""
    width, height = size
    cw, ch = crop_wh
    f0 = int(round((t - cfg["window_before_s"]) * fps))
    n = int(round((t + cfg["window_after_s"]) * fps)) - f0 + 1
    cmd = ["ffmpeg", "-v", "error", *(["-hwaccel", "cuda"] if hw_decode else []),
           "-ss", f"{max(0, f0) / fps:.4f}", "-i", str(video), "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=False).stdout
    size_b = width * height * 3
    images = np.frombuffer(raw, np.uint8)[: len(raw) // size_b * size_b].reshape(-1, height,
                                                                              width, 3)
    pt = path_frames / fps
    known = np.isfinite(path_uv).all(axis=1)
    frames, uv = [], []
    for i, img in enumerate(images):
        f = max(0, f0) + i
        ft = f / fps
        near = known & (np.abs(pt - ft) < 0.2)
        frames.append(f)
        if not near.any():
            uv.append((np.nan, np.nan))
            continue
        u = float(np.interp(ft, pt[near], path_uv[near, 0]))
        v = float(np.interp(ft, pt[near], path_uv[near, 1]))
        x0 = int(np.clip(u - cw / 2, 0, width - cw))
        y0 = int(np.clip(v - ch / 2, 0, height - ch))
        best = None
        for b, s in detect_crop(img, (x0, y0, cw, ch)):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            dist = float(np.hypot(cx - u, cy - v))
            if s >= cfg["min_score"] and dist < cfg["max_jump_px"] and \
                    (best is None or dist < best[0]):
                best = (dist, cx, cy)
        uv.append(best[1:] if best else (np.nan, np.nan))
    return np.array(frames), np.array(uv, float).reshape(-1, 2)


def add_stuff_blocks(rallies: list, contacts: list[Contact], checks: dict[int, dict],
                     cfg: dict) -> int:
    """Insert a block touch after each point-ending attack whose check (keyed by the
    attack's frame) shows the ball coming back off a block; re-judge those outcomes.
    Contacts stay in time order (rally contact indices are shifted). -> blocks added."""
    s = cfg["contacts"]["stuff_block"]
    rows = [r.to_dict() for r in rallies]
    added = 0
    for ci in sorted(candidates(rows, [c.to_dict() for c in contacts]), reverse=True):
        attack = contacts[ci]
        check = checks.get(attack.frame)
        if check is None or check["second_turn_px_s"] < s["min_turn_px_s"]:
            continue
        r = next(r for r in rallies if ci in r.contacts)
        t = attack.t + check["turn_dt_s"]
        block = replace(attack, t=round(t, 4), frame=int(round(t * check["fps"])),
                        team=r.point_winner, track_id=-1, ball_xyz=None,
                        confidence=s["confidence"], serve=False, entry_serve=False,
                        action="block", action_confidence=s["confidence"])
        pos = ci + 1
        contacts.insert(pos, block)
        for x in rallies:                       # indices at or after pos move up by one
            x.contacts = [i + (i >= pos) for i in x.contacts]
            if x.serve_contact is not None and x.serve_contact >= pos:
                x.serve_contact += 1
        r.contacts.insert(r.contacts.index(ci) + 1, pos)
        outcome, conf = point_outcome("block", r.point_winner, r.point_winner, s["confidence"])
        r.outcome, r.outcome_confidence = outcome, round(conf * r.winner_confidence, 3)
        added += 1
    return added
