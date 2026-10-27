"""Score pipeline touches and rallies against hand labels (data/labels/<id>/events.csv).

Only the labelled stretch counts: a prediction is judged only if it falls inside a labelled
rally (padded by `rally_margin_s`), so labelling 10 rallies of a set doesn't turn every
touch in the other 30 into a false alarm.

Touches: predicted touches (plus serves traced back from flight, which have no touch of
their own) are matched one-to-one to labelled contacts within `match_tolerance_s`, closest
first. Floor contacts are not touches; they're scored as rally ends.

Rallies: a labelled rally is found if a predicted rally overlaps it; then the start (serve)
and end errors, and whether the serving team is right. Labels are a continuous run of
rallies, so a predicted rally between the first and last label that overlaps none of them
is a false rally (warm-up balls in a timeout, a ball kicked back over between points).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from setter_eyes.classify.outcome import point_outcome
from setter_eyes.labels.events import RallyLabel

TRACED = -2                     # rallies.json serve_contact when traced back from flight


@dataclass
class TouchMatch:
    label_t: float | None       # None: a false alarm
    pred_t: float | None        # None: a miss
    label_team: str | None = None
    pred_team: str | None = None
    action: str | None = None
    confidence: float | None = None
    source: str = ""            # touch | traced
    rally: int | None = None    # labelled rally number
    pred_action: str | None = None


@dataclass
class RallyMatch:
    rally: int
    label_start: float
    label_end: float
    label_serving: str | None
    pred_index: int | None = None
    pred_start: float | None = None
    pred_end: float | None = None
    pred_serving: str | None = None
    serve_how: str = ""         # seen | traced | inferred | not found
    end_reason: str = ""
    label_outcome: str | None = None
    pred_outcome: str | None = None   # from the pipeline's last touch + the true winner
    label_winner: str | None = None
    pred_winner: str | None = None    # inferred: who serves next (rally/winner.py)
    pred_outcome_inferred: str | None = None   # last touch + the inferred winner


@dataclass
class Evaluation:
    touches: list[TouchMatch] = field(default_factory=list)
    rallies: list[RallyMatch] = field(default_factory=list)
    false_rallies: list[dict] = field(default_factory=list)   # predicted, matching no label

    def summary(self) -> dict:
        matched = [m for m in self.touches if m.label_t is not None and m.pred_t is not None]
        n_label = sum(m.label_t is not None for m in self.touches)
        n_pred = sum(m.pred_t is not None for m in self.touches)
        team_ok = [m.label_team == m.pred_team for m in matched]
        dt = [abs(m.pred_t - m.label_t) for m in matched]
        acted = [m for m in matched if m.pred_action is not None]
        by_action: dict[str, list[int]] = {}
        for m in self.touches:
            if m.label_t is not None:
                hit = by_action.setdefault(m.action or "?", [0, 0])
                hit[0] += m.pred_t is not None
                hit[1] += 1
        found = [r for r in self.rallies if r.pred_index is not None]
        served = [r for r in found if r.serve_how != "not found"]
        return {
            "touches_labelled": n_label,
            "touches_predicted": n_pred,
            "touch_recall": len(matched) / n_label if n_label else None,
            "touch_precision": len(matched) / n_pred if n_pred else None,
            "team_accuracy": float(np.mean(team_ok)) if team_ok else None,
            "median_time_error_s": float(np.median(dt)) if dt else None,
            "recall_by_action": {a: f"{h}/{n}" for a, (h, n) in sorted(by_action.items())},
            "action_accuracy": float(np.mean([m.pred_action == m.action for m in acted]))
            if acted else None,
            "action_accuracy_by_label": {
                a: f"{sum(m.pred_action == a for m in acted if m.action == a)}/"
                   f"{sum(m.action == a for m in acted)}"
                for a in sorted({m.action for m in acted if m.action})},
            "rallies_labelled": len(self.rallies),
            "rallies_found": len(found),
            "false_rallies": len(self.false_rallies),
            "serves_found": len(served),
            "serving_team_correct": sum(r.pred_serving == r.label_serving for r in served),
            "median_start_error_s": _median_abs(r.pred_start - r.label_start for r in served),
            "median_end_error_s": _median_abs(r.pred_end - r.label_end for r in found),
            "outcomes_right": sum(r.pred_outcome == r.label_outcome for r in found
                                  if r.label_outcome and r.label_outcome != "replay"),
            "outcomes_labelled": sum(bool(r.label_outcome) and r.label_outcome != "replay"
                                     for r in found),
            "winners_labelled": sum(bool(r.label_winner) for r in found),
            "winners_inferred": sum(bool(r.label_winner and r.pred_winner) for r in found),
            "winners_right": sum(bool(r.label_winner) and r.pred_winner == r.label_winner
                                 for r in found),
            "outcomes_inferred_right": sum(r.pred_outcome_inferred == r.label_outcome
                                           for r in found if r.label_outcome
                                           and r.label_outcome != "replay"),
        }


def _median_abs(values) -> float | None:
    v = [abs(x) for x in values]
    return float(np.median(v)) if v else None


def evaluate(labels: list[RallyLabel], contacts: list[dict], rallies: list[dict], fps: float,
             cfg: dict) -> Evaluation:
    """labels: parsed events.csv; contacts / rallies: contacts.json / rallies.json lists."""
    e = cfg["evaluate"]
    tol, margin = e["match_tolerance_s"], e["rally_margin_s"]
    windows = [(r.start_frame / fps - margin, r.end_frame / fps + margin, r.rally_number)
               for r in labels]

    def window_of(t: float) -> int | None:
        for lo, hi, n in windows:
            if lo <= t <= hi:
                return n
        return None

    # Labelled side.
    lab = [(c.frame / fps, c.team, c.action, r.rally_number)
           for r in labels for c in r.contacts]
    # Predicted side: touches inside the labelled stretch, plus traced serves.
    pred = [(c["t"], c["team"], c["confidence"], "touch", c.get("action")) for c in contacts
            if c["kind"] == "touch" and window_of(c["t"]) is not None]
    for ra in rallies:
        if ra["serve_contact"] == TRACED and window_of(ra["start_t"]) is not None:
            pred.append((ra["start_t"], ra["serving_team"], ra["confidence"], "traced",
                         "serve"))

    out = Evaluation()
    pairs = _match([p[0] for p in pred], [lb[0] for lb in lab], tol)
    used_p = {i for i, _ in pairs}
    used_l = {j for _, j in pairs}
    for i, j in pairs:
        p, lb = pred[i], lab[j]
        out.touches.append(TouchMatch(lb[0], p[0], lb[1], p[1], lb[2], p[2], p[3], lb[3],
                                      p[4]))
    for j, lb in enumerate(lab):
        if j not in used_l:
            out.touches.append(TouchMatch(lb[0], None, lb[1], None, lb[2], rally=lb[3]))
    for i, p in enumerate(pred):
        if i not in used_p:
            out.touches.append(TouchMatch(None, p[0], None, p[1], None, p[2], p[3],
                                          window_of(p[0]), p[4]))
    out.touches.sort(key=lambda m: m.label_t if m.label_t is not None else m.pred_t)

    for r in labels:
        t0, t1 = r.start_frame / fps, r.end_frame / fps
        m = RallyMatch(r.rally_number, t0, t1, r.serving_team, label_outcome=r.outcome)
        best, overlap = None, 0.0
        for ra in rallies:
            ov = min(t1, ra["end_t"]) - max(t0, ra["start_t"])
            if ov > overlap:
                best, overlap = ra, ov
        if best is not None:
            sc = best["serve_contact"]
            m.pred_index, m.pred_start, m.pred_end = best["index"], best["start_t"], best["end_t"]
            m.pred_serving = best["serving_team"]
            m.serve_how = {TRACED: "traced", -3: "inferred", None: "not found"}.get(sc, "seen")
            m.end_reason = best["end_reason"]
            m.label_winner = r.point_winner
            m.pred_winner = best.get("point_winner")
            m.pred_outcome_inferred = best.get("outcome")
            touches = [contacts[i] for i in best.get("contacts", [])
                       if i < len(contacts) and contacts[i]["kind"] == "touch"]
            last = touches[-1] if touches else {}
            if r.point_winner:
                m.pred_outcome = point_outcome(last.get("action"), last.get("team"),
                                               r.point_winner,
                                               last.get("action_confidence"))[0]
        out.rallies.append(m)
    if labels:
        lo, hi = labels[0].start_frame / fps, labels[-1].end_frame / fps
        spans = [(r.start_frame / fps, r.end_frame / fps) for r in labels]
        out.false_rallies = [
            ra for ra in rallies
            if ra["end_t"] >= lo and ra["start_t"] <= hi
            and not any(min(b, ra["end_t"]) > max(a, ra["start_t"]) for a, b in spans)]
    return out


def _match(pred_t: list[float], label_t: list[float], tol: float) -> list[tuple[int, int]]:
    """One-to-one pairs (pred, label) within tol, minimising total time error."""
    if not pred_t or not label_t:
        return []
    cost = np.abs(np.subtract.outer(np.array(pred_t), np.array(label_t)))
    big = 1e6
    rows, cols = linear_sum_assignment(np.where(cost <= tol, cost, big))
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if cost[i, j] <= tol]
