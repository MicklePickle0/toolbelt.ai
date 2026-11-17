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


def pixel_scale(width: int) -> float:
    """Pixel settings are written for video.reference_width (4K); a video this wide
    scales them by the returned factor (1080p: 0.5)."""
    return width / load("pipeline")["video"]["reference_width"]


def scale_px(section: dict[str, Any], width: int) -> dict[str, Any]:
    """Copy of a config section with its pixel settings scaled to a video `width` wide.

    Pixel settings are keys ending in `_px` or containing `_px_` (a nested section or
    list under such a key is scaled whole); `per_px` keys scale inversely. Other keys
    are copied, nested sections recursively."""
    s = pixel_scale(width)

    def walk(value: Any, factor: float | None) -> Any:
        if isinstance(value, dict):
            return {k: walk(v, factor if factor is not None else _px_factor(k, s))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, factor) for v in value]
        if factor is not None and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            return value * factor
        return value

    return walk(section, None)


def for_video(pipeline: dict[str, Any], width: int) -> dict[str, Any]:
    """The pipeline config with the pixel settings of the per-video stages (detect, track,
    ball, contacts) scaled to a video `width` wide."""
    return {**pipeline, **{k: scale_px(pipeline[k], width)
                           for k in ("detect", "track", "ball", "contacts")}}


def _px_factor(key: str, s: float) -> float | None:
    if "per_px" in key:
        return 1 / s
    if key.endswith("_px") or "_px_" in key:
        return s
    return None


def sample_stride(fps: float) -> int:
    """Every Nth frame for whole-video passes, so they sample at video.sample_hz whatever
    the frame rate (59.94 fps: 2, 30 fps: 1). Settings counted in samples assume this rate."""
    return max(1, round(fps / load("pipeline")["video"]["sample_hz"]))
