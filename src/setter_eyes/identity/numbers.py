"""Jersey numbers from back views (Milestone 7, player identity design steps 2 and 5).

A player seen from behind (pose: their left shoulder on the image's left, shoulders wide
enough, not side on) shows the back number between the shoulders, from just below the
shoulder line to about 60% of the way down to the hips (`back_crop`). PARSeq (a pretrained
scene-text reader, torch hub `baudm/parseq`) reads it, limited to the video's roster: each
roster number's log P(its digits, then the end), so "33" or "3" for a 13 in this font still
lands on 13 (`NumberReader`). Checked by eye on 59 readable back views of pakmen-black_set2
(1080p, 2026-10-31): 76% right; of the reads taking >= 80% of the roster posterior, 86%.

A read counts when PARSeq finds it plausible as text at all (log P of its best roster
number >= `min_log_p`: blurred, partial and side views otherwise all read as "1") and its
number takes `min_share` of the roster posterior.

A track is NOT one player throughout: the tracker hands an id from one player to another
between points (pakmen-black_set2 track 0: the libero #6 for 3 s, then #14 for 16 s, then
#17, every read right). So each track is cut into stretches where its reads agree
(`stretches`: a new number needs `switch_reads` reads in a row), and a touch takes the
number of the stretch it falls in (`number_at`), with confidence top / (all + vote_prior)
of that stretch's votes, lower the further the touch is from its nearest read. The
binding holds through the stretch, also when the number is hidden.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12


def back_crop(kp: np.ndarray, score: np.ndarray, cfg: dict) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the back number of a player facing away, or None (facing the
    camera, side on, or the shoulders or hips not found). cfg: identity.numbers."""
    pts = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if min(score[i] for i in pts) < cfg["keypoint_min_score"]:
        return None
    ls, rs, lh, rh = (kp[i] for i in pts)
    width = rs[0] - ls[0]                      # > 0: their left shoulder on the image's left
    y_sh = (ls[1] + rs[1]) / 2
    torso = (lh[1] + rh[1]) / 2 - y_sh
    if torso <= 0 or width < cfg["back_min_shoulder_ratio"] * torso:
        return None
    x0, x1 = ls[0] - cfg["crop_side"] * width, rs[0] + cfg["crop_side"] * width
    y0, y1 = y_sh + cfg["crop_top"] * torso, y_sh + cfg["crop_bottom"] * torso
    return int(max(0, x0)), int(max(0, y0)), int(x1), int(y1)


class NumberReader:
    """PARSeq limited to a roster's numbers."""

    def __init__(self, roster: list[int], device: str = "cuda"):
        import torch  # heavy imports only when reading numbers

        self.torch = torch
        self.device = device
        self.roster = [str(n) for n in roster]
        self.model = torch.hub.load("baudm/parseq", "parseq", pretrained=True,
                                    trust_repo=True).eval().to(device)
        index = {s: i for i, s in enumerate(self.model.tokenizer._itos)}
        self.codes = [[index[ch] for ch in s] for s in self.roster]

    def _batch(self, crops: list[np.ndarray]):
        ims = []
        for c in crops:
            h, w = c.shape[:2]
            side = max(0, (4 * h - w) // 2)      # pad to the reader's 4:1 text line
            c = cv2.copyMakeBorder(c, 0, 0, side, side, cv2.BORDER_REPLICATE)
            c = cv2.resize(c, (128, 32), interpolation=cv2.INTER_AREA)
            ims.append(self.torch.from_numpy(c[:, :, ::-1].copy()).permute(2, 0, 1))
        x = self.torch.stack(ims).float().to(self.device) / 255
        return (x - 0.5) / 0.5

    def scores(self, crops: list[np.ndarray], batch: int = 128) -> np.ndarray:
        """(len(crops), len(roster)) log P(number) for each crop."""
        out = []
        for k in range(0, len(crops), batch):
            with self.torch.no_grad():
                lp = self.model(self._batch(crops[k:k + batch])).log_softmax(-1).cpu().numpy()
            out.append(np.stack([
                [sum(row[t, ch] for t, ch in enumerate(code)) + row[len(code), 0]
                 for code in self.codes] for row in lp]))
        return np.concatenate(out) if out else np.zeros((0, len(self.roster)))


def shares(log_scores: np.ndarray) -> np.ndarray:
    """Posterior over the roster numbers per read (rows sum to 1)."""
    p = np.exp(log_scores - log_scores.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


@dataclass
class Stretch:
    track_id: int
    first: int                  # frames covered (to halfway to the neighbouring stretches)
    last: int
    number: int
    confidence: float
    reads: np.ndarray           # frames of its reads

    def to_dict(self) -> dict:
        return {"track_id": self.track_id, "first": self.first, "last": self.last,
                "number": self.number, "confidence": self.confidence,
                "reads": len(self.reads)}


def counted(log_scores: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(mask of the reads that count, index of each read's roster number)."""
    if not len(log_scores):
        return np.zeros(0, bool), np.zeros(0, int)
    p = shares(log_scores)
    top = p.argmax(axis=1)
    ok = (log_scores.max(axis=1) >= cfg["min_log_p"]) & \
        (p[np.arange(len(p)), top] >= cfg["min_share"])
    return ok, top


def stretches(track_ids: np.ndarray, frames: np.ndarray, log_scores: np.ndarray,
              roster: list[int], cfg: dict) -> list[Stretch]:
    """Each track cut where its counted reads change number for `switch_reads` in a row."""
    ok, top = counted(log_scores, cfg)
    share = shares(log_scores) if len(log_scores) else np.zeros((0, len(roster)))
    out: list[Stretch] = []
    k_switch = cfg["switch_reads"]
    for tid in np.unique(track_ids[ok]):
        rows = np.flatnonzero(ok & (track_ids == tid))
        rows = rows[np.argsort(frames[rows], kind="stable")]
        groups: list[list[int]] = [[rows[0]]]
        for j, r in enumerate(rows[1:], start=1):
            votes: dict[int, float] = defaultdict(float)
            for q in groups[-1]:
                votes[top[q]] += share[q, top[q]]
            lead = max(votes, key=votes.get)
            ahead = rows[j:j + k_switch]
            if top[r] != lead and len(ahead) == k_switch and (top[ahead] == top[r]).all():
                groups.append([r])
            else:
                groups[-1].append(r)
        for g in groups:
            votes = defaultdict(float)
            for q in g:
                votes[top[q]] += share[q, top[q]]
            lead = max(votes, key=votes.get)
            conf = votes[lead] / (sum(votes.values()) + cfg["vote_prior"])
            reads = np.sort(frames[g]).astype(int)
            out.append(Stretch(int(tid), int(reads[0]), int(reads[-1]), roster[lead],
                               round(float(conf), 3), reads))
    # The first stretch of a track reaches back to the track's start, the others start
    # halfway between their first read and the previous stretch's last.
    by_track: dict[int, list[Stretch]] = defaultdict(list)
    for s in out:
        by_track[s.track_id].append(s)
    for ss in by_track.values():
        ss.sort(key=lambda s: s.first)
        ss[0].first = 0
        for a, b in zip(ss, ss[1:], strict=False):
            mid = (int(a.reads[-1]) + int(b.reads[0])) // 2
            a.last, b.first = mid, mid + 1
        ss[-1].last = 10**9
    return out


class NumberIndex:
    """number_at(track, frame) over a video's stretches."""

    def __init__(self, items: list[Stretch], fps: float, cfg: dict):
        self.by_track: dict[int, list[Stretch]] = defaultdict(list)
        for s in items:
            self.by_track[s.track_id].append(s)
        self.fps, self.cfg = fps, cfg

    def number_at(self, track_id: int, frame: int) -> tuple[int, float] | None:
        for s in self.by_track.get(int(track_id), []):
            if s.first <= frame <= s.last:
                gap_s = float(np.min(np.abs(s.reads - frame))) / self.fps
                fade = float(np.exp(-max(0.0, gap_s - self.cfg["fade_after_s"])
                                    / self.cfg["fade_s"]))
                return s.number, round(s.confidence * fade, 3)
        return None
