"""Review UI data layer (Milestone 6): read rallies and touches from the event DB, save a
review. No Streamlit here, so it is testable; the app is ui/review_app.py.

Saving a rally's review, in one transaction:
  - point winner (us / them, or None for a replayed point), its source 'user', the
    outcome (suggested from the last touch, overridable), the serving team, the start and
    end times;
  - touch edits: changed time / team / action, deleted touches, added touches (confidence
    1.0); touches before the start or after the end are removed;
  - every changed field is logged in `corrections` (table, row, field, old, new, author):
    human corrections are training data (CLAUDE.md);
  - the rally is marked reviewed, and so are its touches when the reviewer says they are
    all right. Re-running the pipeline keeps reviewed rallies (db/pipeline.py).

A point the pipeline split in two is put back together with merge_rallies() (the
neighbour's touches move over, the time span covers both, the neighbour goes).
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from setter_eyes.classify.outcome import point_outcome
from setter_eyes.db.pipeline import renumber

REPLAY = "replay"
NOT_A_RALLY = "not_a_rally"     # a false rally (warm-up, another court): kept as reviewed so
                                # re-runs don't bring it back; its touches are false ones


@dataclass
class TouchEdit:
    id: int | None              # None: a touch the reviewer added
    t: float                    # seconds into the video
    team: str | None
    action: str | None
    delete: bool = False


def games(conn: sqlite3.Connection) -> list[dict]:
    q = """SELECT g.id, g.video_key, g.venue, g.fps, g.video_path, s.id AS set_id,
                  COUNT(r.id) AS rallies, COALESCE(SUM(r.reviewed), 0) AS reviewed
           FROM games g JOIN sets s ON s.game_id = g.id LEFT JOIN rallies r ON r.set_id = s.id
           GROUP BY g.id, s.id ORDER BY g.video_key"""
    return [dict(r) for r in conn.execute(q)]


def rallies(conn: sqlite3.Connection, set_id: int) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT id, rally_number, start_t, end_t, segment_confidence, serving_team, "
        "point_winner, outcome, reviewed FROM rallies WHERE set_id = ? ORDER BY start_t",
        (set_id,))]


def touches(conn: sqlite3.Connection, rally_id: int) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT id, t, team, action, contact_confidence, action_confidence, reviewed "
        "FROM contacts WHERE rally_id = ? ORDER BY t", (rally_id,))]


def score(rows: list[dict]) -> tuple[int, int]:
    """(us, them) from the reviewed winners so far (replays score nothing)."""
    us = sum(r["reviewed"] and r["point_winner"] == "us" for r in rows)
    them = sum(r["reviewed"] and r["point_winner"] == "them" for r in rows)
    return us, them


def review_queue(rows: list[dict]) -> list[dict]:
    """Unreviewed rallies, least confident first (CLAUDE.md: low confidence -> review)."""
    return sorted((r for r in rows if not r["reviewed"]),
                  key=lambda r: (r["segment_confidence"] or 0.0, r["start_t"]))


def suggest_outcome(edits: list[TouchEdit], winner: str | None,
                    end_t: float | None = None) -> str | None:
    """The outcome rule (classify/outcome.py) on the edited touches up to the rally's end;
    None without a winner, `replay` for a replayed point."""
    if winner in (REPLAY, NOT_A_RALLY):
        return winner
    if winner is None:
        return None
    kept = sorted((e for e in edits if not e.delete and (end_t is None or e.t <= end_t)),
                  key=lambda e: e.t)
    last = kept[-1] if kept else None
    return point_outcome(last.action if last else None, last.team if last else None,
                         winner)[0]


def _log(conn, table: str, row_id: int, field: str, old, new, author: str) -> None:
    if old == new:
        return
    conn.execute("INSERT INTO corrections (table_name, row_id, field, old_value, new_value, "
                 "author) VALUES (?, ?, ?, ?, ?, ?)",
                 (table, row_id, field, None if old is None else str(old),
                  None if new is None else str(new), author))


def save_review(conn: sqlite3.Connection, rally_id: int, fps: float, winner: str | None,
                outcome: str | None, serving_team: str | None, end_t: float,
                edits: list[TouchEdit], touches_checked: bool,
                author: str = "owner", start_t: float | None = None) -> None:
    """winner: 'us' | 'them' | REPLAY | NOT_A_RALLY. start_t: None keeps the start."""
    point_winner = winner if winner in ("us", "them") else None
    if winner in (REPLAY, NOT_A_RALLY):
        outcome = winner
    with conn:
        old = dict(conn.execute("SELECT * FROM rallies WHERE id = ?", (rally_id,)).fetchone())
        before = {r["id"]: r for r in touches(conn, rally_id)}
        if start_t is None:
            start_t = old["start_t"]
        for e in edits:
            if e.t > end_t or e.t < start_t - 0.01:
                e.delete = True                   # outside the rally: not one of its touches
            if e.id is None:
                if e.delete:
                    continue
                new_id = conn.execute(
                    "INSERT INTO contacts (rally_id, t, frame, contact_confidence, team, "
                    "action, action_confidence, reviewed) VALUES (?, ?, ?, 1.0, ?, ?, 1.0, ?)",
                    (rally_id, e.t, round(e.t * fps), e.team, e.action, int(touches_checked)),
                ).lastrowid
                _log(conn, "contacts", new_id, "insert", None, f"{e.t:.2f} {e.team} {e.action}",
                     author)
                continue
            b = before.get(e.id)
            if b is None:
                continue
            if e.delete:
                _log(conn, "contacts", e.id, "delete", f"{b['t']:.2f} {b['team']} {b['action']}",
                     None, author)
                conn.execute("DELETE FROM contacts WHERE id = ?", (e.id,))
                continue
            for field, new in (("t", round(e.t, 3)), ("team", e.team), ("action", e.action)):
                old_v = round(b[field], 3) if field == "t" else b[field]
                _log(conn, "contacts", e.id, field, old_v, new, author)
            # A confirmed touch, or one the reviewer changed, is certain from now on.
            touch_conf = 1.0 if touches_checked else b["contact_confidence"]
            action_conf = 1.0 if touches_checked or e.action != b["action"] else \
                b["action_confidence"]
            conn.execute("UPDATE contacts SET t = ?, frame = ?, team = ?, action = ?, "
                         "contact_confidence = ?, action_confidence = ?, reviewed = ? "
                         "WHERE id = ?",
                         (e.t, round(e.t * fps), e.team, e.action, touch_conf, action_conf,
                          int(touches_checked), e.id))
        for field, new in (("point_winner", point_winner), ("outcome", outcome),
                           ("serving_team", serving_team), ("start_t", round(start_t, 3)),
                           ("end_t", round(end_t, 3))):
            old_v = round(old[field], 3) if field in ("start_t", "end_t") else old[field]
            _log(conn, "rallies", rally_id, field, old_v, new, author)
        conn.execute("UPDATE rallies SET point_winner = ?, point_winner_source = ?, "
                     "outcome = ?, outcome_confidence = 1.0, serving_team = ?, start_t = ?, "
                     "start_frame = ?, end_t = ?, end_frame = ?, reviewed = 1 WHERE id = ?",
                     (point_winner, "user" if point_winner else None, outcome, serving_team,
                      start_t, round(start_t * fps), end_t, round(end_t * fps), rally_id))


def neighbours(rows: list[dict], rally_id: int) -> tuple[dict | None, dict | None]:
    """(previous, next) rally in time order."""
    i = next(k for k, r in enumerate(rows) if r["id"] == rally_id)
    return (rows[i - 1] if i > 0 else None), (rows[i + 1] if i + 1 < len(rows) else None)


def merge_rallies(conn: sqlite3.Connection, keep_id: int, other_id: int,
                  author: str = "owner") -> None:
    """One point the pipeline split in two: `other`'s touches move to `keep`, whose time
    span grows to cover both; `other` is removed. Logged in corrections; the merged rally
    still needs its review (winner etc.) saved."""
    with conn:
        k = dict(conn.execute("SELECT * FROM rallies WHERE id = ?", (keep_id,)).fetchone())
        o = dict(conn.execute("SELECT * FROM rallies WHERE id = ?", (other_id,)).fetchone())
        first, last = (k, o) if k["start_t"] <= o["start_t"] else (o, k)
        conn.execute("UPDATE contacts SET rally_id = ? WHERE rally_id = ?", (keep_id, other_id))
        conn.execute("UPDATE rallies SET start_t = ?, start_frame = ?, end_t = ?, end_frame = ?,"
                     " serving_team = ? WHERE id = ?",
                     (first["start_t"], first["start_frame"], last["end_t"], last["end_frame"],
                      first["serving_team"], keep_id))
        _log(conn, "rallies", keep_id, "merge", f"rally {k['rally_number']}",
             f"+ rally {o['rally_number']} ({o['start_t']:.2f}-{o['end_t']:.2f})", author)
        conn.execute("DELETE FROM rallies WHERE id = ?", (other_id,))
        renumber(conn, k["set_id"])


def rally_clip(src: Path, out: Path, t0: float, t1: float, width: int = 1280) -> Path:
    """A short H.264 clip of [t0, t1] from src (the proxy when there is one), cached."""
    if out.exists():
        return out
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t0):.3f}", "-i", str(src),
           "-t", f"{t1 - max(0.0, t0):.3f}", "-vf", f"scale={width}:-2", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(out)]
    subprocess.run(cmd, check=True)
    return out
