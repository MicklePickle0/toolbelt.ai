"""Match format per game (roadmap E1, owner 2026-11-16): how many sets, points per set, win by.

Most OVA matches are best of 3 (25, 25, 15); 18U and high-school playoffs can be best of 5
(25, 25, 25, 25, 15). Formats vary by tournament, so nothing is hardcoded: presets live in
config/pipeline.yaml (`match.formats`), and each game picks one or a custom format.

Human work, so a text file in git like the folders and rosters: data/match_formats.yaml
  games:        game (folders.py: the game a set is filed under) -> preset name, or
                {points: [25, 25, 15], win_by: 2, cap_over: null}
  set_numbers:  video key -> which set of the match it is, when the key doesn't say
                (a key ending in _set3 is set 3; otherwise set 1)
A game not in it uses `match.default`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from setter_eyes import config


class MatchFormatError(ValueError):
    pass


@dataclass(frozen=True)
class MatchFormat:
    points: tuple[int, ...]            # points to win each set, in order
    win_by: int = 2
    cap_over: int | None = None        # cap: target + this wins outright (2: 25 -> 27, 15 -> 17)
    preset: str | None = None          # the preset's name; None for a custom format

    @property
    def best_of(self) -> int:
        return len(self.points)

    def target(self, set_number: int) -> int:
        """Points to win set `set_number` (1-based); past the last set, the last set's."""
        return self.points[min(max(set_number, 1), len(self.points)) - 1]

    def set_over(self, a: int, b: int, set_number: int) -> bool:
        """Has one side won the set at a-b?"""
        hi, lo = max(a, b), min(a, b)
        if self.cap_over is not None and hi >= self.target(set_number) + self.cap_over:
            return True
        return hi >= self.target(set_number) and hi - lo >= self.win_by

    def label(self) -> str:
        pts = "-".join(str(p) for p in self.points)
        cap = f", cap +{self.cap_over}" if self.cap_over else ""
        return f"best of {self.best_of} ({pts}), win by {self.win_by}{cap}"

    def to_dict(self) -> dict:
        return {"preset": self.preset, "points": list(self.points), "win_by": self.win_by,
                "cap_over": self.cap_over, "best_of": self.best_of, "label": self.label()}


def presets(cfg: dict | None = None) -> dict[str, MatchFormat]:
    m = (cfg or config.load("pipeline"))["match"]
    return {name: _custom(spec, name) for name, spec in m["formats"].items()}


def default(cfg: dict | None = None) -> MatchFormat:
    cfg = cfg or config.load("pipeline")
    return presets(cfg)[cfg["match"]["default"]]


def _custom(spec: dict, preset: str | None = None) -> MatchFormat:
    try:
        points = tuple(int(p) for p in spec["points"])
        win_by = int(spec.get("win_by", 2))
        cap = spec.get("cap_over")
        cap = None if cap in (None, "", 0) else int(cap)
    except (KeyError, TypeError, ValueError) as e:
        raise MatchFormatError(f"a match format needs points: [25, 25, 15] (got {spec!r})") from e
    if not points or len(points) % 2 == 0 or any(not 1 <= p <= 99 for p in points):
        raise MatchFormatError("points: one entry per set, an odd number of sets (best of 1, 3, "
                               f"5...), each 1-99 (got {list(points)})")
    if not 1 <= win_by <= 5:
        raise MatchFormatError(f"win_by must be 1-5 (got {win_by})")
    if cap is not None and not 1 <= cap <= 20:
        raise MatchFormatError(f"cap_over must be 1-20 points over the target (got {cap})")
    return MatchFormat(points, win_by, cap, preset)


def parse(value: object, cfg: dict | None = None) -> MatchFormat:
    """A preset name or a {points, win_by, cap_over} mapping -> MatchFormat."""
    if isinstance(value, str):
        p = presets(cfg)
        if value not in p:
            raise MatchFormatError(f"unknown match format {value!r}: one of {sorted(p)} or "
                                   "{points: [...], win_by: 2, cap_over: null}")
        return p[value]
    if isinstance(value, dict):
        return _custom(value)
    raise MatchFormatError(f"a match format is a preset name or a mapping (got {value!r})")


# ---- the per-game file --------------------------------------------------------------------
def load(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {"games": {}, "set_numbers": {}}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {"games": dict(raw.get("games") or {}),
            "set_numbers": {str(k): int(v) for k, v in (raw.get("set_numbers") or {}).items()}}


def save(path: Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    body = {"games": dict(sorted(data.get("games", {}).items())),
            "set_numbers": dict(sorted(data.get("set_numbers", {}).items()))}
    Path(path).write_text(
        "# Match format per game (match.py): a preset from config/pipeline.yaml `match.formats`\n"
        "# or {points: [25, 25, 15], win_by: 2, cap_over: null}; set_numbers: video -> set of the "
        "match.\n" + yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")


def format_for(game: str, data: dict, cfg: dict | None = None) -> MatchFormat:
    spec = data.get("games", {}).get(game)
    return default(cfg) if spec is None else parse(spec, cfg)


_SET_RE = re.compile(r"_set(\d+)")


def set_number(video_key: str, data: dict) -> int:
    """Which set of its match a video is: set_numbers, else the key's _setN, else 1."""
    if video_key in data.get("set_numbers", {}):
        return int(data["set_numbers"][video_key])
    m = _SET_RE.search(video_key)
    return int(m.group(1)) if m else 1
