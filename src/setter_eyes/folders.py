"""Folders: sets grouped into games, games into tournaments (owner, 2026-11-01).

Human work, so a text file in git like the rosters: data/folders.yaml maps a video key to
{tournament, game}. A video not in it is in no tournament, and its game is its key up to
"_set" (pakmen-black_set2 -> pakmen-black), so sets of one match group without any setup.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

UNFILED = "No tournament"


def default_game(key: str) -> str:
    return re.sub(r"_set\d+.*$", "", key)


def load(path: Path) -> dict[str, dict[str, str]]:
    if not Path(path).exists():
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): {"tournament": str(v.get("tournament") or ""),
                     "game": str(v.get("game") or "")} for k, v in raw.items()}


def save(path: Path, folders: dict[str, dict[str, str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    body = {k: {"tournament": v["tournament"], "game": v["game"]}
            for k, v in sorted(folders.items())}
    Path(path).write_text("# Video key -> tournament and game (the web app's Stats page).\n"
                          + yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")


def folder_of(key: str, folders: dict[str, dict[str, str]]) -> dict[str, str]:
    f = folders.get(key, {})
    return {"tournament": f.get("tournament") or UNFILED,
            "game": f.get("game") or default_game(key)}


def tree(keys: list[str], folders: dict[str, dict[str, str]]) -> list[dict]:
    """[{tournament, games: [{game, videos: [key, ...]}]}], sorted; unfiled last."""
    by: dict[str, dict[str, list[str]]] = {}
    for k in keys:
        f = folder_of(k, folders)
        by.setdefault(f["tournament"], {}).setdefault(f["game"], []).append(k)
    order = sorted(by, key=lambda t: (t == UNFILED, t.lower()))
    return [{"tournament": t, "games": [{"game": g, "videos": sorted(v)}
                                        for g, v in sorted(by[t].items())]} for t in order]
