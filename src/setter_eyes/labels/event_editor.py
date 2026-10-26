"""Keyboard event labelling: the logic behind scripts/label_events.py (no UI here).

Events are rows of events.csv (docs/labelling.md). Keys, at the current frame:
  s            rally_start
  u / t        a touch by us / them, then the action key (ACTION_KEYS); for our
               touches the jersey (digits + Enter, Enter alone skips), and for our
               receive the pass score (0-4, Enter skips)
  e            rally_end, then the point winner (u / t), then the outcome
               (OUTCOME_KEYS, Enter skips); or r instead of a winner: the point is
               replayed (a ball rolled on, a referee decision), nobody scores
  Esc          cancels a half-entered event
  z            undo the last event added;  x  delete the event at this frame
Every change is saved straight away.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from setter_eyes.labels.events import parse_time

COLUMNS = ["set", "frame", "t", "type", "team", "jersey", "action", "pass_score",
           "point_winner", "outcome", "note"]
ACTION_KEYS = {"1": "serve", "2": "receive", "3": "set", "4": "attack", "5": "block",
               "6": "dig", "7": "free_ball", "8": "dump", "9": "overpass", "0": "coverage"}
OUTCOME_KEYS = {"k": "kill", "a": "attack_error", "r": "reception_error", "c": "ace",
                "v": "service_error", "b": "block_point", "h": "ball_handling_error",
                "o": "other"}
TEAM_KEYS = {"u": "us", "t": "them"}
ENTER, ESC, BACKSPACE = "\r", "\x1b", "\b"


def clock(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


@dataclass
class Event:
    frame: int
    type: str
    fields: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        f = self.fields
        if self.type == "contact":
            who = f.get("team", "?") + (f" #{f['jersey']}" if f.get("jersey") else "")
            score = f" ({f['pass_score']})" if f.get("pass_score") else ""
            return f"{who} {f.get('action', '?')}{score}"
        if self.type == "rally_end":
            if f.get("outcome") == "replay":
                return "END: replayed, no point"
            how = f" {f['outcome']}" if f.get("outcome") else ""
            return f"END: point {f.get('point_winner', '?')}{how}"
        return "RALLY START"


class EventEditor:
    def __init__(self, path: Path, fps: float, set_number: int = 1):
        self.path, self.fps, self.set_number = path, fps, set_number
        self.events: list[Event] = []
        self.added: list[Event] = []          # for undo
        self.pending: dict | None = None      # a half-entered event
        self.message = ""
        if path.exists():
            self._load()

    # --- file -------------------------------------------------------------------------
    def _load(self) -> None:
        with self.path.open(encoding="utf-8-sig", newline="") as f:
            for raw in csv.DictReader(f):
                row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
                if not row.get("type") or row["type"].startswith("#"):
                    continue
                if row.get("frame"):
                    frame = int(row["frame"])
                elif row.get("t"):
                    frame = round(parse_time(row["t"]) * self.fps)
                else:
                    continue
                fields = {k: v for k, v in row.items()
                          if k not in ("frame", "t", "type", "set") and v}
                self.events.append(Event(frame, row["type"].lower(), fields))
        self.events.sort(key=lambda e: e.frame)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for e in self.events:
                w.writerow({"set": self.set_number, "frame": e.frame,
                            "t": clock(e.frame / self.fps), "type": e.type, **e.fields})
        tmp.replace(self.path)

    # --- queries ----------------------------------------------------------------------
    def rally_at(self, frame: int) -> list[Event] | None:
        """Events of the rally open or ending at `frame` (from its start), else None."""
        start = None
        for i, e in enumerate(self.events):
            if e.frame > frame:
                break
            if e.type == "rally_start":
                start = i
            elif e.type == "rally_end" and e.frame < frame:
                start = None
        if start is None:
            return None
        out = []
        for e in self.events[start:]:
            if e.type == "rally_start" and out:
                break
            out.append(e)
            if e.type == "rally_end":
                break
        return out

    def at(self, frame: int, within: int = 0) -> list[Event]:
        return [e for e in self.events if abs(e.frame - frame) <= within]

    def neighbour(self, frame: int, forward: bool) -> int | None:
        frames = [e.frame for e in self.events]
        cands = [f for f in frames if f > frame] if forward else [f for f in frames if f < frame]
        if not cands:
            return None
        return min(cands) if forward else max(cands)

    @property
    def prompt(self) -> str:
        p = self.pending
        if p is None:
            return ""
        if p["step"] == "action":
            keys = " ".join(f"{k}={v}" for k, v in ACTION_KEYS.items())
            return f"{p['team']} touch - action: {keys}   (Esc cancels)"
        if p["step"] == "jersey":
            return f"jersey number: {p['buffer']}_   (Enter; Enter alone = unknown)"
        if p["step"] == "pass":
            return "pass score 0-4   (Enter skips)"
        if p["step"] == "winner":
            return "point winner: u = us, t = them, r = replayed (no point)   (Esc cancels)"
        keys = " ".join(f"{k}={v}" for k, v in OUTCOME_KEYS.items())
        return f"outcome: {keys}   (Enter skips)"

    # --- keys -------------------------------------------------------------------------
    def key(self, ch: str, frame: int) -> bool:
        """Handle an event key at `frame`. True if it was used (not a navigation key)."""
        if self.pending is not None:
            self._pending_key(ch)
            return True
        if ch == "s":
            self._add(Event(frame, "rally_start"))
        elif ch in TEAM_KEYS:
            self.pending = {"step": "action", "frame": frame, "team": TEAM_KEYS[ch]}
        elif ch == "e":
            self.pending = {"step": "winner", "frame": frame}
        elif ch == "z":
            if self.added:
                e = self.added.pop()
                if e in self.events:
                    self.events.remove(e)
                    self.save()
                self.message = f"undone: {e.describe()}"
        elif ch == "x":
            here = self.at(frame)
            if here:
                self.events.remove(here[-1])
                self.save()
                self.message = f"deleted: {here[-1].describe()}"
        else:
            return False
        return True

    def _pending_key(self, ch: str) -> None:
        p = self.pending
        if ch == ESC:
            self.pending, self.message = None, "cancelled"
            return
        step = p["step"]
        if step == "action" and ch in ACTION_KEYS:
            p["action"] = ACTION_KEYS[ch]
            if p["team"] == "us":
                p["step"], p["buffer"] = "jersey", ""
            else:
                self._finish_contact()
        elif step == "jersey":
            if ch.isdigit() and len(p["buffer"]) < 2:
                p["buffer"] += ch
            elif ch == BACKSPACE:
                p["buffer"] = p["buffer"][:-1]
            elif ch == ENTER:
                p["jersey"] = p["buffer"]
                if p["action"] == "receive":
                    p["step"] = "pass"
                else:
                    self._finish_contact()
        elif step == "pass" and (ch in "01234" or ch == ENTER):
            p["pass_score"] = "" if ch == ENTER else ch
            self._finish_contact()
        elif step == "winner" and ch == "r":
            self.pending = None
            self._add(Event(p["frame"], "rally_end", {"outcome": "replay"}))
        elif step == "winner" and ch in TEAM_KEYS:
            p["winner"], p["step"] = TEAM_KEYS[ch], "outcome"
        elif step == "outcome" and (ch in OUTCOME_KEYS or ch == ENTER):
            fields = {"point_winner": p["winner"]}
            if ch != ENTER:
                fields["outcome"] = OUTCOME_KEYS[ch]
            self.pending = None
            self._add(Event(p["frame"], "rally_end", fields))

    def _finish_contact(self) -> None:
        p = self.pending
        fields = {"team": p["team"], "action": p["action"]}
        for k in ("jersey", "pass_score"):
            if p.get(k):
                fields[k] = p[k]
        self.pending = None
        self._add(Event(p["frame"], "contact", fields))

    def _add(self, e: Event) -> None:
        i = len([x for x in self.events if x.frame <= e.frame])
        self.events.insert(i, e)
        self.added.append(e)
        self.save()
        self.message = f"added at {clock(e.frame / self.fps)}: {e.describe()}"
