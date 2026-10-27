"""Who won each point, from who serves next; then the running score and the point outcome.

The winner of a rally serves the next one, so the next rally's serving team is this rally's
winner (labelled sets: 79 of 79). That is enough to judge the HOW of a point without seeing
why it ended: an attacker swung and their side scored -> kill, even if the ball only nicked
a blocker's fingers on its way out. The owner's key press in the review UI (and scoreboard
OCR, when visible) overrides it; this is an inference, stored with its confidence and
`point_winner_source = next_serve`.

Confidence: the next rally's own confidence (its serve seen, traced or inferred), lowered
when the gap to it is long (`suspect_gap_s`: a missed rally in between would hand this one
the wrong winner; a timeout looks the same, so it is a doubt, not a rule). The last rally
has no next serve: winner unknown.
"""

from __future__ import annotations

from setter_eyes.classify.outcome import point_outcome
from setter_eyes.contacts.events import Contact
from setter_eyes.rally.segment import Rally


def infer_winners(rallies: list[Rally], contacts: list[Contact], cfg: dict) -> None:
    """Sets point_winner, winner_confidence, score_us/score_them, outcome and
    outcome_confidence on each rally (in time order)."""
    w = cfg["rally"]["winner"]
    us = them = 0
    for r, nxt in zip(rallies, [*rallies[1:], None], strict=True):
        if nxt is None or nxt.serving_team is None:
            r.point_winner, r.winner_confidence = None, 0.0
        else:
            conf = nxt.confidence
            if nxt.start_t - r.end_t > w["suspect_gap_s"]:
                conf *= w["suspect_factor"]
            r.point_winner, r.winner_confidence = nxt.serving_team, round(min(1.0, conf), 3)
        us += r.point_winner == "us"
        them += r.point_winner == "them"
        r.score_us, r.score_them = us, them
        if r.point_winner is None:
            r.outcome, r.outcome_confidence = None, 0.0
            continue
        touches = [contacts[i] for i in r.contacts if contacts[i].kind == "touch"]
        last = touches[-1] if touches else None
        outcome, conf = point_outcome(last.action if last else None,
                                      last.team if last else None, r.point_winner,
                                      last.action_confidence if last else None)
        r.outcome, r.outcome_confidence = outcome, round(conf * r.winner_confidence, 3)
