"""The data manifest: which videos exist, their metadata and label files.

See config/manifest.example.yaml and docs/labelling.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from setter_eyes import config
from setter_eyes.labels.issues import Issues, LabelError

SPLITS = ("train", "valid", "test")
_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def venue_key(venue: str) -> str:
    """Normalised venue name, so "Queen's ARC" and "queens arc" are the same venue."""
    return re.sub(r"[^a-z0-9]", "", venue.lower())


@dataclass(frozen=True)
class RosterEntry:
    number: int
    name: str
    libero: bool = False


@dataclass(frozen=True)
class VideoEntry:
    key: str
    path: Path
    venue: str = ""
    net_height_m: float = 2.43
    net_height_preset: str | None = None  # None: custom height
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
        venue = str(v.get("venue") or "").strip()
        if not venue_key(venue):
            issues.error(f"{key}: missing 'venue' (gym or tournament site)")
        try:
            net_height_m, preset = _net_height(v.get("net_height"))
        except ValueError as e:
            issues.error(f"{key}: {e}")
            net_height_m, preset = 0.0, None
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
            venue=venue,
            net_height_m=net_height_m,
            net_height_preset=preset,
            opponent=v.get("opponent"),
            played_on=str(v["played_on"]) if v.get("played_on") else None,
            split=split,
            events=_label_path(root, key, labels, "events", "events.csv"),
            boxes=_label_path(root, key, labels, "boxes", "boxes.coco.json"),
            lineups=lineups,
        ))

    if issues.errors:
        raise LabelError(issues.errors)
    return Manifest(path=path, team=str(team), roster=roster, videos=videos)


def _net_height(value: object) -> tuple[float, str | None]:
    """A preset name from config/court.yaml or a custom height in metres -> (metres, preset)."""
    net = config.load("court")["net"]
    presets: dict[str, float] = net["height_presets"]
    if isinstance(value, str) and value in presets:
        return float(presets[value]), value
    lo, hi = net["custom_height_range"]
    if isinstance(value, (int, float)) and not isinstance(value, bool) and lo <= value <= hi:
        return float(value), None
    raise ValueError(f"net_height must be one of {sorted(presets)} or metres in "
                     f"[{lo}, {hi}] (got {value!r})")


def _label_path(root: Path, key: str, labels: dict, field: str, default_name: str) -> Path | None:
    """An explicit labels.<field> path, else labels/<id>/<default_name> if that file exists."""
    if labels.get(field):
        return root / labels[field]
    default = root / "labels" / key / default_name
    return default if default.exists() else None
