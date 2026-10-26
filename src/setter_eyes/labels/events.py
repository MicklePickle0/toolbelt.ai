"""Hand-labelled rally/contact events (CSV). Format: docs/labelling.md.

One row per event. Rows may be in any order; they are sorted by frame.

    set,frame,t,type,team,jersey,action,pass_score,point_winner,outcome,note

type is rally_start | contact | rally_end. Give either frame or t (seconds,
"m:ss.sss" or "h:mm:ss.sss"); t is converted with the video's fps.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from setter_eyes import config
from setter_eyes.labels.issues import Issues

EVENT_TYPES = ("rally_start", "contact", "rally_end")
TEAMS = ("us", "them")
REPLAY = "replay"           # outcome of a replayed point: no winner, left out of stats
_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d*)?)$")


@dataclass(frozen=True)
class ContactLabel:
    frame: int
    team: str
    action: str | None
    jersey: int | None = None
    pass_score: int | None = None
    note: str | None = None
    line: int | None = None


@dataclass
class RallyLabel:
    set_number: int
    rally_number: int
    start_frame: int
    end_frame: int
    point_winner: str | None
    outcome: str | None = None
    contacts: list[ContactLabel] = field(default_factory=list)
    line: int | None = None

    @property
    def replayed(self) -> bool:
        return self.outcome == REPLAY

    @property
    def serving_team(self) -> str | None:
        for c in self.contacts:
            if c.action == "serve":
                return c.team
        return None


@dataclass(frozen=True)
class _Row:
    line: int
    set_number: int
    frame: int
    type: str
    fields: dict[str, str]


def parse_time(text: str) -> float:
    """Seconds from '12.5', '1:02.25' or '1:01:02.25'."""
    m = _TIME_RE.match(text.strip())
    if not m:
        raise ValueError(f"bad time {text!r}")
    a, b, s = m.groups()
    if b is not None:
        return int(a) * 3600 + int(b) * 60 + float(s)
    if a is not None:
        return int(a) * 60 + float(s)
    return float(s)


def _vocab() -> dict[str, Any]:
    return config.load("labels")


def parse_events(path: str | Path, fps: float, frame_count: int | None = None,
                 roster_numbers: set[int] | None = None,
                 max_contacts_per_side: int | None = None) -> tuple[list[RallyLabel], Issues]:
    """Parse and validate an events CSV into rallies. Returns (rallies, issues).

    Rallies are only trustworthy if issues.errors is empty.
    """
    path = Path(path)
    issues = Issues(str(path))
    vocab = _vocab()
    actions, outcomes = set(vocab["actions"]), set(vocab["outcomes"])
    if max_contacts_per_side is None:
        max_contacts_per_side = config.load("pipeline")["contacts"]["max_contacts_per_side"]

    rows = _read_rows(path, fps, frame_count, issues)
    rows.sort(key=lambda r: (r.frame, r.line))  # same frame: keep file order
    for prev, cur in zip(rows, rows[1:], strict=False):
        if cur.set_number < prev.set_number:
            issues.error(f"set {cur.set_number} at frame {cur.frame} comes after set "
                         f"{prev.set_number} at frame {prev.frame}", cur.line)

    rallies: list[RallyLabel] = []
    current: RallyLabel | None = None
    next_number: dict[int, int] = {}
    for row in rows:
        f = row.fields
        if row.type == "rally_start":
            if current is not None:
                issues.error(f"rally_start while rally from line {current.line} is still open "
                             "(missing rally_end?)", row.line)
            number = next_number.get(row.set_number, 1)
            if f.get("rally"):
                try:
                    number = int(f["rally"])
                except ValueError:
                    issues.error(f"rally must be an integer, got {f['rally']!r}", row.line)
            if any(r.set_number == row.set_number and r.rally_number == number for r in rallies):
                issues.error(f"duplicate rally {number} in set {row.set_number}", row.line)
            next_number[row.set_number] = number + 1
            current = RallyLabel(row.set_number, number, row.frame, row.frame, None,
                                 line=row.line)
        elif current is None:
            issues.error(f"{row.type} outside a rally (no open rally_start)", row.line)
        elif row.set_number != current.set_number:
            issues.error(f"{row.type} is in set {row.set_number} but its rally started in set "
                         f"{current.set_number}", row.line)
        elif row.type == "contact":
            contact = _parse_contact(row, actions, roster_numbers, issues)
            if contact:
                current.contacts.append(contact)
        else:  # rally_end
            current.end_frame = row.frame
            winner = f.get("point_winner", "").lower() or None
            outcome = f.get("outcome", "").lower() or None
            if outcome is not None and outcome not in outcomes:
                issues.error(f"unknown outcome {outcome!r} (config/labels.yaml)", row.line)
            if outcome == REPLAY:
                if winner is not None:
                    issues.warning("point_winner ignored on a replay (nobody gets the point)",
                                   row.line)
                winner = None
            elif winner not in TEAMS:
                issues.error(f"rally_end needs point_winner us|them (or outcome {REPLAY}), "
                             f"got {winner!r}", row.line)
            current.point_winner = winner if winner in TEAMS else None
            current.outcome = outcome
            _check_rally(current, max_contacts_per_side, issues)
            rallies.append(current)
            current = None

    if current is not None:
        issues.error("file ends inside a rally (missing rally_end)", current.line)
    return rallies, issues


def _read_rows(path: Path, fps: float, frame_count: int | None, issues: Issues) -> list[_Row]:
    rows: list[_Row] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            issues.error("empty file")
            return rows
        headers = {h.strip().lower() for h in reader.fieldnames if h}
        for required in ("type",):
            if required not in headers:
                issues.error(f"missing column {required!r}")
                return rows
        if not headers & {"frame", "t"}:
            issues.error("need a 'frame' or 't' column")
            return rows

        set_number = 1
        for line, raw in enumerate(reader, start=2):
            fields = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            if not any(fields.values()) or fields.get("type", "").startswith("#"):
                continue  # blank or commented-out row
            if fields.get("set"):
                try:
                    set_number = int(fields["set"])
                except ValueError:
                    issues.error(f"set must be an integer, got {fields['set']!r}", line)
                    continue
            etype = fields.get("type", "").lower()
            if etype not in EVENT_TYPES:
                issues.error(f"type must be one of {EVENT_TYPES}, got {etype!r}", line)
                continue
            try:
                if fields.get("frame"):
                    frame = int(fields["frame"])
                elif fields.get("t"):
                    frame = round(parse_time(fields["t"]) * fps)
                else:
                    issues.error("row has neither frame nor t", line)
                    continue
            except ValueError as e:
                issues.error(f"bad frame/t: {e}", line)
                continue
            if frame < 0 or (frame_count is not None and frame >= frame_count):
                issues.error(f"frame {frame} outside video (0..{frame_count})", line)
                continue
            rows.append(_Row(line, set_number, frame, etype, fields))
    return rows


def _parse_contact(row: _Row, actions: set[str], roster_numbers: set[int] | None,
                   issues: Issues) -> ContactLabel | None:
    f = row.fields
    team = f.get("team", "").lower()
    if team not in TEAMS:
        issues.error(f"contact needs team us|them, got {team!r}", row.line)
        return None
    action = f.get("action", "").lower() or None
    if action is None:
        issues.warning("contact without action", row.line)
    elif action not in actions:
        issues.error(f"unknown action {action!r} (config/labels.yaml)", row.line)
        return None

    jersey = None
    if f.get("jersey"):
        try:
            jersey = int(f["jersey"])
        except ValueError:
            issues.error(f"jersey must be an integer, got {f['jersey']!r}", row.line)
        if team == "them":
            issues.warning("jersey on an opponent contact is ignored", row.line)
            jersey = None
        elif jersey is not None and roster_numbers and jersey not in roster_numbers:
            issues.warning(f"jersey {jersey} is not in the roster", row.line)

    pass_score = None
    if f.get("pass_score"):
        try:
            pass_score = int(f["pass_score"])
        except ValueError:
            issues.error(f"pass_score must be 0-4, got {f['pass_score']!r}", row.line)
        if pass_score is not None and not 0 <= pass_score <= 4:
            issues.error(f"pass_score must be 0-4, got {pass_score}", row.line)
            pass_score = None
        if pass_score is not None and (action != "receive" or team != "us"):
            issues.error("pass_score only applies to our serve receive (action receive, team us)",
                         row.line)
            pass_score = None
    return ContactLabel(row.frame, team, action, jersey, pass_score, f.get("note") or None,
                        row.line)


def _check_rally(rally: RallyLabel, max_per_side: int, issues: Issues) -> None:
    """Soft checks on the contact chain: warnings, since real rallies get weird."""
    c = rally.contacts
    if not c:
        issues.warning("rally has no contacts", rally.line)
        return
    if c[0].action not in (None, "serve"):
        issues.warning(f"first contact is {c[0].action!r}, expected serve", c[0].line)
    for prev, cur in zip(c, c[1:], strict=False):
        if cur.action == "receive" and not (prev.action == "serve" and prev.team != cur.team):
            issues.warning("receive should directly follow the opponent's serve "
                           "(use dig for defence)", cur.line)
    run = 0
    for prev, cur in zip([None, *c], c, strict=False):
        if cur.action == "block":
            run = 0  # a block doesn't count toward the three contacts
            continue
        run = run + 1 if prev is not None and prev.team == cur.team else 1
        if run > max_per_side:
            issues.warning(f"{run} consecutive contacts by {cur.team} (max {max_per_side})",
                           cur.line)
