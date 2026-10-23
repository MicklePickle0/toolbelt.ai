"""Train/valid/test assignment BY VIDEO, never by frame.

Videos pinned in the manifest keep their split. The rest are bucketed by a
hash of their key, so adding a video never moves an existing one to another
split (which would leak it into a model trained on the old split).
"""

from __future__ import annotations

import hashlib

from setter_eyes import config
from setter_eyes.labels.manifest import SPLITS, Manifest


def _unit(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_for(key: str, ratios: dict[str, float]) -> str:
    total = sum(ratios.values())
    u, acc = _unit(key), 0.0
    for name in SPLITS:
        acc += ratios.get(name, 0.0) / total
        if u < acc:
            return name
    return SPLITS[0]


def assign_splits(manifest: Manifest, ratios: dict[str, float] | None = None) -> dict[str, str]:
    """video key -> split name (train | valid | test)."""
    ratios = ratios or config.load("pipeline")["datasets"]["split_ratios"]
    if unknown := set(ratios) - set(SPLITS):
        raise ValueError(f"unknown split names {sorted(unknown)}; use {SPLITS}")
    return {v.key: v.split or split_for(v.key, ratios) for v in manifest.videos}
