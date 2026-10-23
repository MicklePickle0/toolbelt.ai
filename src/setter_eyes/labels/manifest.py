"""The data manifest: which videos exist, their metadata and label files.

See config/manifest.example.yaml and docs/labelling.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from setter_eyes.labels.issues import Issues, LabelError

SPLITS = ("train", "valid", "test")
_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class RosterEntry:
    number: int
    name: str
    libero: bool = False


@dataclass(frozen=True)
class VideoEntry:
    key: str
    path: Path
    division: str = "men"
    opponent: str | None = None
    played_on: str | None = None
    split: str | None = None
    events: Path | None = None
    boxes: Path | None = None
    lineups: dict[int, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class Manifest:
    path: Path
    team: str
    roster: list[RosterEntry]
    videos: list[VideoEntry]

    def video(self, key: str) -> VideoEntry:
        for v in self.videos:
            if v.key == key:
                return v
        raise KeyError(f"no video {key!r} in {self.path}")

    @property
    def roster_numbers(self) -> set[int]:
        return {r.number for r in self.roster}


def load_manifest(path: str | Path) -> Manifest:
    """Parse and validate a manifest. Raises LabelError listing every problem."""
    path = Path(path)
    issues = Issues(str(path))
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    root = path.parent

    team = raw.get("team")
    if not team:
        issues.error("missing 'team' (our team's name)")

    roster: list[RosterEntry] = []
    for i, r in enumerate(raw.get("roster") or []):
        try:
            roster.append(RosterEntry(int(r["number"]), str(r["name"]), bool(r.get("libero"))))
        except (KeyError, TypeError, ValueError):
            issues.error(f"roster[{i}] needs integer 'number' and 'name': {r!r}")
    numbers = [r.number for r in roster]
    for n in {n for n in numbers if numbers.count(n) > 1}:
        issues.error(f"roster has jersey {n} more than once")

    videos: list[VideoEntry] = []
    seen: set[str] = set()
    for i, v in enumerate(raw.get("videos") or []):
        where = f"videos[{i}]"
        key = str(v.get("id", ""))
        if not _KEY_RE.match(key):
            issues.error(f"{where}: 'id' must be letters, digits, _ or - (got {key!r})")
            continue
        if key in seen:
            issues.error(f"{where}: duplicate id {key!r}")
        seen.add(key)
        if not v.get("path"):
            issues.error(f"{key}: missing 'path'")
            continue
        division = v.get("division", "men")
        if division not in ("men", "women"):
            issues.error(f"{key}: division must be men or women (got {division!r})")
        split = v.get("split")
        if split is not None and split not in SPLITS:
            issues.error(f"{key}: split must be one of {SPLITS} (got {split!r})")
        labels = v.get("labels") or {}

        lineups: dict[int, list[int]] = {}
        for set_no, lineup in (v.get("lineups") or {}).items():
            try:
                lineup = [int(n) for n in lineup]
            except (TypeError, ValueError):
                issues.error(f"{key}: lineup for set {set_no} must be jersey numbers")
                continue
            if len(lineup) != 6 or len(set(lineup)) != 6:
                issues.error(f"{key}: lineup for set {set_no} needs six distinct jerseys")
            if roster and (unknown := set(lineup) - set(numbers)):
                issues.error(f"{key}: lineup for set {set_no} has jerseys not in roster: "
                             f"{sorted(unknown)}")
            lineups[int(set_no)] = lineup

        videos.append(VideoEntry(
            key=key,
            path=root / v["path"],
            division=division,
            opponent=v.get("opponent"),
            played_on=str(v["played_on"]) if v.get("played_on") else None,
            split=split,
            events=root / labels["events"] if labels.get("events") else None,
            boxes=root / labels["boxes"] if labels.get("boxes") else None,
            lineups=lineups,
        ))

    if issues.errors:
        raise LabelError(issues.errors)
    return Manifest(path=path, team=str(team), roster=roster, videos=videos)
