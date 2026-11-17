"""Rosters: our teams' players (jersey number, name, height, position), several of them, each
applied to the videos it played in.

Rosters are human work, so the truth is text files in git, like the labels:
  data/rosters/<slug>.yaml        one roster: name, season, notes, players
  data/rosters/assignments.yaml   video key -> roster slug
The event DB gets a copy (sync_db) so stats can be queried with SQL: each roster is a
`teams` row (teams.roster_slug) with its `players`, and an assigned video's game points at
that team. Players keep their ids across syncs (matched by jersey number), so touches
already credited to a player stay credited when the roster is edited.

A video without an assigned roster uses the manifest's team, as before.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ASSIGNMENTS = "assignments.yaml"
_SLUG = re.compile(r"[^a-z0-9]+")
_FEET = re.compile(r"^\s*(\d)\s*(?:'|ft|’)\s*(\d{1,2}(?:\.\d+)?)?\s*(?:\"|in|”|'')?\s*$")
HEIGHT_RANGE_CM = (120.0, 230.0)


class RosterError(ValueError):
    pass


@dataclass
class RosterPlayer:
    number: int
    name: str
    height_cm: float | None = None
    position: str | None = None          # free text: S, OH, MB, OPP, L, DS ...
    libero: bool = False


@dataclass
class Roster:
    slug: str
    name: str
    season: str | None = None
    notes: str | None = None
    players: list[RosterPlayer] = field(default_factory=list)

    def to_yaml(self) -> dict:
        d = {"name": self.name, "season": self.season, "notes": self.notes,
             "players": [{k: v for k, v in asdict(p).items() if v not in (None, False)}
                         for p in sorted(self.players, key=lambda p: p.number)]}
        return {k: v for k, v in d.items() if v is not None}


def slugify(name: str) -> str:
    slug = _SLUG.sub("-", name.lower()).strip("-")
    if not slug:
        raise RosterError("a roster needs a name")
    return slug


def parse_height(text: str | float | None) -> float | None:
    """'188', '188 cm', '6\\'2"', '6 ft 2', '6\\'' -> centimetres (None if blank)."""
    if text is None or (isinstance(text, str) and not text.strip()):
        return None
    if isinstance(text, int | float):
        cm = float(text)
    else:
        t = text.strip().lower()
        m = _FEET.match(t)
        if m:
            cm = (int(m.group(1)) * 12 + float(m.group(2) or 0)) * 2.54
        else:
            try:
                cm = float(t.removesuffix("cm").strip())
            except ValueError as e:
                raise RosterError(f"height {text!r}: give cm (188) or feet and inches "
                                  "(6'2\")") from e
    if not HEIGHT_RANGE_CM[0] <= cm <= HEIGHT_RANGE_CM[1]:
        raise RosterError(f"height {text!r} ({cm:.0f} cm) looks wrong")
    return round(cm, 1)


def validate(roster: Roster) -> None:
    seen = set()
    for p in roster.players:
        if not 0 <= p.number <= 99:
            raise RosterError(f"jersey {p.number}: numbers are 0-99")
        if p.number in seen:
            raise RosterError(f"jersey {p.number} is on the roster twice")
        if not p.name.strip():
            raise RosterError(f"jersey {p.number} needs a name")
        seen.add(p.number)


# ---- files ---------------------------------------------------------------------------------
def load_all(directory: Path) -> dict[str, Roster]:
    out = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        if path.name == ASSIGNMENTS:
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        players = [RosterPlayer(int(p["number"]), str(p["name"]), parse_height(p.get("height_cm")),
                                p.get("position"), bool(p.get("libero", False)))
                   for p in raw.get("players") or []]
        r = Roster(path.stem, raw.get("name") or path.stem, raw.get("season"), raw.get("notes"),
                   players)
        validate(r)
        out[r.slug] = r
    return out


def save(directory: Path, roster: Roster) -> Path:
    validate(roster)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{roster.slug}.yaml"
    path.write_text(yaml.safe_dump(roster.to_yaml(), sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def load_assignments(directory: Path) -> dict[str, str]:
    path = Path(directory) / ASSIGNMENTS
    if not path.exists():
        return {}
    return {str(k): str(v) for k, v in (yaml.safe_load(path.read_text(encoding="utf-8"))
                                        or {}).items()}


def save_assignments(directory: Path, assignments: dict[str, str]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text = ("# Which roster each video was played with (video key -> roster file name).\n"
            + yaml.safe_dump(dict(sorted(assignments.items())), sort_keys=False))
    (directory / ASSIGNMENTS).write_text(text, encoding="utf-8")


def delete(directory: Path, slug: str) -> None:
    """Remove a roster's file; refused while videos use it."""
    used = [k for k, s in load_assignments(directory).items() if s == slug]
    if used:
        raise RosterError(f"roster {slug!r} is applied to {len(used)} video(s): "
                          f"{', '.join(sorted(used))}; take it off them first")
    (Path(directory) / f"{slug}.yaml").unlink(missing_ok=True)


# ---- event DB copy -------------------------------------------------------------------------
def _team_id(conn: sqlite3.Connection, roster: Roster) -> int:
    row = conn.execute("SELECT id FROM teams WHERE roster_slug = ?", (roster.slug,)).fetchone()
    if row is None:
        # A team of that name may exist without a roster link (e.g. the manifest's team).
        row = conn.execute("SELECT id FROM teams WHERE name = ? AND roster_slug IS NULL",
                           (roster.name,)).fetchone()
    if row is None:
        name, n = roster.name, 2
        while conn.execute("SELECT 1 FROM teams WHERE name = ?", (name,)).fetchone():
            name, n = f"{roster.name} ({n})", n + 1
        return conn.execute("INSERT INTO teams (name, roster_slug, season) VALUES (?, ?, ?)",
                            (name, roster.slug, roster.season)).lastrowid
    conn.execute("UPDATE teams SET roster_slug = ?, season = ? WHERE id = ?",
                 (roster.slug, roster.season, row[0]))
    return row[0]


def _move_game(conn: sqlite3.Connection, game_id: int, team_id: int) -> None:
    """Point a game at another team; touches credited to a player follow by jersey."""
    old = conn.execute("SELECT team_id FROM games WHERE id = ?", (game_id,)).fetchone()[0]
    if old == team_id:
        return
    new = dict(conn.execute("SELECT jersey_number, id FROM players WHERE team_id = ?",
                            (team_id,)).fetchall())
    rows = conn.execute(
        "SELECT c.id, p.jersey_number FROM contacts c JOIN players p ON c.player_id = p.id "
        "JOIN rallies r ON c.rally_id = r.id JOIN sets s ON r.set_id = s.id "
        "WHERE s.game_id = ?", (game_id,)).fetchall()
    for cid, jersey in rows:
        pid = new.get(jersey)
        if pid is None:                     # not on the new roster: keep the touch, by number
            pid = conn.execute("INSERT INTO players (team_id, jersey_number, name) "
                               "VALUES (?, ?, ?)", (team_id, jersey, f"#{jersey}")).lastrowid
            new[jersey] = pid
        conn.execute("UPDATE contacts SET player_id = ? WHERE id = ?", (pid, cid))
    conn.execute("UPDATE games SET team_id = ? WHERE id = ?", (team_id, game_id))


def sync_db(conn: sqlite3.Connection, directory: Path) -> dict[str, int]:
    """Copy the roster files into the DB: teams + players, and each assigned video's game
    onto its roster's team. Returns {roster slug: team id}."""
    rosters = load_all(directory)
    teams = {}
    with conn:
        for r in rosters.values():
            tid = teams[r.slug] = _team_id(conn, r)
            have = dict(conn.execute("SELECT jersey_number, id FROM players WHERE team_id = ?",
                                     (tid,)).fetchall())
            for p in r.players:
                values = (p.name, int(p.libero), p.height_cm, p.position)
                if p.number in have:
                    conn.execute("UPDATE players SET name = ?, is_libero = ?, height_cm = ?, "
                                 "position = ? WHERE id = ?", (*values, have[p.number]))
                else:
                    conn.execute("INSERT INTO players (name, is_libero, height_cm, position, "
                                 "team_id, jersey_number) VALUES (?, ?, ?, ?, ?, ?)",
                                 (*values, tid, p.number))
            listed = {p.number for p in r.players}
            for jersey, pid in have.items():   # taken off the roster: gone unless credited
                if jersey not in listed and not conn.execute(
                        "SELECT 1 FROM contacts WHERE player_id = ? LIMIT 1", (pid,)).fetchone():
                    conn.execute("DELETE FROM players WHERE id = ?", (pid,))
        for key, slug in load_assignments(directory).items():
            if slug not in teams:
                continue
            row = conn.execute("SELECT id FROM games WHERE video_key = ?", (key,)).fetchone()
            if row is not None:
                _move_game(conn, row[0], teams[slug])
    return teams


def team_for_video(conn: sqlite3.Connection, directory: Path | None, video_key: str) -> int | None:
    """The team id of the roster assigned to this video (synced first), or None."""
    if directory is None:
        return None
    slug = load_assignments(directory).get(video_key)
    if slug is None:
        return None
    return sync_db(conn, directory).get(slug)
