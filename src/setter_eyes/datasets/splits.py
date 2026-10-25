"""Train/valid/test assignment by venue (default) or by video, never by frame.

With split_by: venue, every video from one venue lands in the same split, so
valid/test measure how the model does in a gym it has never seen. Pinning
one video of a venue pins the whole venue. Unpinned groups are bucketed by a
hash of their name, so adding a video or venue never moves an existing one to
another split (which would leak it into a model trained on the old split).
"""

from __future__ import annotations

import hashlib

from setter_eyes import config
from setter_eyes.labels.issues import Issue, LabelError
from setter_eyes.labels.manifest import SPLITS, Manifest, VideoEntry, venue_key

SPLIT_BY = ("venue", "video")


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


def group_of(entry: VideoEntry, split_by: str) -> str:
    return f"venue:{venue_key(entry.venue)}" if split_by == "venue" else entry.key


def assign_splits(manifest: Manifest, ratios: dict[str, float] | None = None,
                  split_by: str | None = None) -> dict[str, str]:
    """video key -> split name (train | valid | test)."""
    ds = config.load("pipeline")["datasets"]
    ratios = ratios or ds["split_ratios"]
    split_by = split_by or ds.get("split_by", "venue")
    if unknown := set(ratios) - set(SPLITS):
        raise ValueError(f"unknown split names {sorted(unknown)}; use {SPLITS}")
    if split_by not in SPLIT_BY:
        raise ValueError(f"split_by must be one of {SPLIT_BY} (got {split_by!r})")

    pins: dict[str, dict[str, list[str]]] = {}  # group -> split -> pinned video keys
    for v in manifest.videos:
        if v.split:
            pins.setdefault(group_of(v, split_by), {}).setdefault(v.split, []).append(v.key)
    conflicts = [
        Issue("error", str(manifest.path),
              f"venue {group.removeprefix('venue:')!r} is pinned to several splits "
              f"({'; '.join(f'{s}: {sorted(keys)}' for s, keys in sorted(by_split.items()))}); "
              "all videos from one venue must share a split")
        for group, by_split in pins.items() if len(by_split) > 1
    ]
    if conflicts:
        raise LabelError(conflicts)

    out = {}
    for v in manifest.videos:
        group = group_of(v, split_by)
        pinned = next(iter(pins[group]), None) if group in pins else None
        out[v.key] = pinned or split_for(group, ratios)
    return out
