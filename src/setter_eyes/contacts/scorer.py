"""Learned touch scorer, trained on hand labels (data/labels/<id>/events.csv).

Two small gradient-boosting models (scikit-learn) over contacts/features.py:
  - touch: is this candidate a real touch? Target: the one candidate matched to each
    labelled touch (closest within `match_tolerance_s`); its neighbours are negatives, so
    the model learns to pick one break per touch.
  - team: on real touches, was it ours?
Decoding keeps the most likely candidate within any `nms_s` window, above `threshold`.
Validation is grouped by rally (no rally in both train and test); that is still within
the training videos, so a held-out labelled video is the real test.
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier

from setter_eyes.contacts.events import _merge, find_contacts
from setter_eyes.contacts.features import NAMES, candidate_features


def touch_targets(cand_t: np.ndarray, label_t: np.ndarray, tol: float) -> tuple[np.ndarray,
                                                                                np.ndarray]:
    """-> (is_touch (N,) 0/1, matched label index (N,) or -1), one candidate per label."""
    y = np.zeros(len(cand_t), int)
    which = np.full(len(cand_t), -1)
    if len(cand_t) and len(label_t):
        cost = np.abs(np.subtract.outer(cand_t, label_t))
        rows, cols = linear_sum_assignment(np.where(cost <= tol, cost, 1e6))
        for a, b in zip(rows, cols, strict=True):
            if cost[a, b] <= tol:
                y[a], which[a] = 1, b
    return y, which


def _model(cfg: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
        max_leaf_nodes=cfg["max_leaf_nodes"], min_samples_leaf=cfg["min_samples_leaf"],
        l2_regularization=cfg["l2_regularization"], random_state=0)


@dataclass
class TouchScorer:
    touch: HistGradientBoostingClassifier
    team: HistGradientBoostingClassifier
    trained_on: list[str]
    features: list[str] = field(default_factory=lambda: list(NAMES))
    trained_at: str = ""

    @classmethod
    def fit(cls, X: np.ndarray, is_touch: np.ndarray, is_ours: np.ndarray,
            trained_on: list[str], cfg: dict) -> TouchScorer:
        """is_ours: 1/0 for real touches, -1 elsewhere (ignored by the team model)."""
        touch = _model(cfg["model"]).fit(X, is_touch)
        real = is_ours >= 0
        team = _model(cfg["team_model"]).fit(X[real], is_ours[real])
        return cls(touch, team, trained_on, list(NAMES),
                   datetime.now(UTC).isoformat(timespec="seconds"))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (P(touch), P(ours)) per candidate."""
        if self.features != list(NAMES):
            raise ValueError("scorer was trained on other features; retrain it "
                             "(scripts/train_touch_scorer.py)")
        if not len(X):
            return np.zeros(0), np.zeros(0)
        return self.touch.predict_proba(X)[:, 1], self.team.predict_proba(X)[:, 1]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))
        return path

    @staticmethod
    def load(path: Path) -> TouchScorer:
        return pickle.loads(path.read_bytes())


def decode(times: np.ndarray, p_touch: np.ndarray, threshold: float, nms_s: float) -> list[int]:
    """Indices kept: most likely first, none within nms_s of a kept one; in time order."""
    keep: list[int] = []
    for j in np.argsort(-p_touch):
        if p_touch[j] < threshold:
            break
        if all(abs(times[j] - times[k]) >= nms_s for k in keep):
            keep.append(int(j))
    return sorted(keep)


def raw_candidates(path, arcs, g, fps, people, roles, cfg: dict):
    """Every flight break, before any threshold or merging (contacts/events.py)."""
    raw = copy.deepcopy(cfg)
    raw["contacts"]["min_confidence"] = 0.0
    raw["contacts"]["merge_window_s"] = 0.0
    return find_contacts(path, arcs, g, fps, people, roles, raw)


def scored_contacts(path, arcs, g, fps, people, roles, cfg: dict, scorer: TouchScorer):
    """Contacts with touches chosen and teamed by the learned scorer; floor contacts as
    the heuristic finds them."""
    c, s = cfg["contacts"], cfg["contacts"]["scorer"]
    cands = raw_candidates(path, arcs, g, fps, people, roles, cfg)
    touches = [x for x in cands if x.kind == "touch"]
    floors = _merge([x for x in cands if x.kind == "floor"
                     and x.confidence >= c["min_confidence"]], c["merge_window_s"])
    X = candidate_features(touches, path, people, fps, c["overhead_reach_bodies"])
    p_touch, p_ours = scorer.predict(X)
    times = np.array([x.t for x in touches])
    kept = []
    for j in decode(times, p_touch, s["threshold"], s["nms_s"]):
        x = touches[j]
        x.confidence = round(float(p_touch[j]), 3)
        x.team = "us" if p_ours[j] >= 0.5 else "them"
        x.p_ours = round(float(p_ours[j]), 3)
        kept.append(x)
    return sorted(kept + floors, key=lambda x: x.t)
