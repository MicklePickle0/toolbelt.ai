"""Rotation state machine: who stands where, and who serves, in each rally of a set.

The starting lineup is six jerseys in rotation order: lineup[0] starts in position 1 (the
server), lineup[1] in position 2, and so on. We rotate once each time we win the serve
back (a rally they served, followed by one we serve), so after k rotations position p
holds lineup[(k + p - 1) % 6] and lineup[k % 6] serves. A set we receive first starts
with lineup[1] serving, after the first side-out.

The serving-team sequence drives it: from the labels (exact) or from the pipeline (each
rally's serving team carries the rally's confidence, so the state's confidence falls with
every uncertain side-out since the last anchor). A serve whose jersey is known (labels,
later number reading) anchors the state:
- the expected server: confidence back to 1;
- a jersey not on court: a substitution into the server's slot (the slot keeps it);
- a jersey on court in another slot: a missed or false side-out, or a wrong lineup; the
  rotation is re-synced to that server and flagged.

`rotation` is 1 + (rotations since the start) % 6: rotation 1 is the starting lineup.
The libero (who replaces back-row players and, here, never serves) is not modelled yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RallyRotation:
    rotation: int                                       # 1-6, 1 = the starting lineup
    positions: dict[int, int]                           # court position 1-6 -> jersey
    server: int | None                                  # our server's jersey, if we served
    confidence: float
    flags: list[str] = field(default_factory=list)      # substitution, resync, unknown_serve


def follow_rotation(lineup: list[int], serving_teams: list[str | None], cfg: dict,
                    confidences: list[float] | None = None,
                    servers: list[int | None] | None = None) -> list[RallyRotation]:
    """One RallyRotation per rally, in time order. serving_teams: 'us' / 'them' / None
    (unknown); confidences: how sure each rally's serving team is (default 1);
    servers: our server's jersey where known (None elsewhere)."""
    if len(lineup) != 6 or len(set(lineup)) != 6:
        raise ValueError(f"a lineup needs six distinct jerseys, got {lineup}")
    w = cfg["identity"]["rotation"]
    n = len(serving_teams)
    confidences = confidences or [1.0] * n
    servers = servers or [None] * n
    slots = list(lineup)
    turns, conf, prev = 0, 1.0, None
    out: list[RallyRotation] = []
    for team, team_conf, jersey in zip(serving_teams, confidences, servers, strict=True):
        flags: list[str] = []
        if team is None:
            conf *= w["unknown_serve_factor"]
            flags.append("unknown_serve")
        else:
            if team == "us" and prev == "them":
                turns += 1
                conf *= team_conf
            prev = team
        if team == "us" and jersey is not None:
            if jersey == slots[turns % 6]:
                conf = 1.0
            elif jersey in slots:
                # Nearest rotation (either way) that puts this jersey at the service line.
                shift = (slots.index(jersey) - turns) % 6
                turns += shift if shift <= 3 else shift - 6
                conf = 1.0
                flags.append("resync")
            else:
                slots[turns % 6] = jersey
                conf = 1.0
                flags.append("substitution")
        positions = {p: slots[(turns + p - 1) % 6] for p in range(1, 7)}
        out.append(RallyRotation(
            rotation=turns % 6 + 1, positions=positions,
            server=positions[1] if team == "us" else None,
            confidence=round(conf, 3), flags=flags))
    return out


def suggest_lineup(serving_teams: list[str | None], servers: list[int | None]
                   ) -> tuple[list[int | None], list[str]]:
    """Starting lineup implied by who served our serves (labelled jerseys), and notes on
    anything that doesn't fit one lineup. Slots no one served from stay None (e.g. a set
    too short to go all the way round)."""
    slots: list[int | None] = [None] * 6
    notes: list[str] = []
    turns, prev = 0, None
    for i, (team, jersey) in enumerate(zip(serving_teams, servers, strict=True), start=1):
        if team is None:
            continue
        if team == "us" and prev == "them":
            turns += 1
        prev = team
        if team != "us" or jersey is None:
            continue
        k = turns % 6
        if slots[k] == jersey:
            continue
        if slots[k] is None and jersey not in slots:
            slots[k] = jersey
        elif slots[k] is None:
            notes.append(f"rally {i}: #{jersey} served from slot {k + 1}, but served from "
                         f"slot {slots.index(jersey) + 1} before (missed or false side-out)")
        else:
            notes.append(f"rally {i}: #{jersey} served from slot {k + 1}, where #{slots[k]} "
                         "served before (substitution or missed side-out)")
    return slots, notes
