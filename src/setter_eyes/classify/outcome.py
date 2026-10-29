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
rally_outcome() adds the chain rule: a ball sent over that the other side can't control on
its first touch is a kill.
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


# A ball sent over that the other side can't control on its first touch is a kill for the
# player who sent it (owner, 2026-10-28), whether it was spiked or not.
SENT_OVER = ("attack", "free_ball", "overpass", "dump")
_NOT_A_FAILED_FIRST_TOUCH = ("block", "serve", "receive")


def rally_outcome(touches: list[tuple[str | None, str | None, float | None]],
                  point_winner: str) -> tuple[str, float]:
    """The outcome from the whole touch chain: (team, action, action_confidence) per touch
    in time order. point_outcome() on the last touch, except that the losing side's single
    failed first touch (a dig it couldn't keep in play) after the winner sent the ball over
    makes that ball a kill. A failed receive stays a reception error / ace, and a touched
    block a kill as before."""
    if not touches:
        return point_outcome(None, None, point_winner)
    team, action, conf = touches[-1]
    if team is not None and team != point_winner and action not in _NOT_A_FAILED_FIRST_TOUCH:
        k = len(touches) - 1
        while k > 0 and touches[k - 1][0] == team:
            k -= 1
        before = touches[k - 1] if k > 0 else None
        if len(touches) - k == 1 and before is not None and before[0] == point_winner and \
                before[1] in SENT_OVER:
            c = before[2] if before[2] is not None else 0.5
            return "kill", round(c, 3)
    return point_outcome(action, team, point_winner, conf)
