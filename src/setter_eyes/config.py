"""Load YAML config from config/. Thresholds live there, never in code."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@cache
def load(name: str) -> dict[str, Any]:
    """Load config/<name>.yaml, e.g. load("pass_scale")."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(relative: str | Path) -> Path:
    """Resolve a path from pipeline.yaml relative to the repo root."""
    return REPO_ROOT / relative
