"""Point outcome, the HOW of a point (CLAUDE.md step 9), once the rally is complete.

`point_winner` comes from the review UI or the labels, else it is inferred from who serves
next (rally/winner.py). Given it, the outcome follows from the rally's last touch and whether that
touch's side won, matching how the owner labels (titan-rise_set1):
  last touch   its side won   its side lost
  serve        ace            service_error
  attack       kill           attack_error
  block        block_point    kill            (the attack beat the block)
  receive      other          reception_error (shank)
  anything     other          ball_handling_error (dig, set, free ball, coverage the
  else                                          side couldn't keep in play)
No touch at all (a serve never seen, the ball lost) -> other, at low confidence.
Confidence: the last touch's action confidence (the rules are certain given it).
"""

from __future__ import annotations

_WON = {"serve": "ace", "attack": "kill", "block": "block_point"}
_LOST = {"serve": "service_error", "attack": "attack_error", "block": "kill",
         "receive": "reception_error"}


def point_outcome(last_action: str | None, last_team: str | None, point_winner: str,
                  action_confidence: float | None = None) -> tuple[str, float]:
    if last_action is None or last_team is None:
        return "other", 0.2
    won = last_team == point_winner
    outcome = _WON.get(last_action, "other") if won else \
        _LOST.get(last_action, "ball_handling_error")
    conf = action_confidence if action_confidence is not None else 0.5
    return outcome, round(conf if outcome != "other" else conf * 0.5, 3)
