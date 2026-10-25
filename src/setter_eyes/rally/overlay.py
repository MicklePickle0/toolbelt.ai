"""Rally banner: rally number, serving team, how the serve was found and how it ended."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class RallyOverlay:
    def __init__(self, cache_dir: Path):
        c = json.loads((cache_dir / "contacts.json").read_text(encoding="utf-8"))
        r = json.loads((cache_dir / "rallies.json").read_text(encoding="utf-8"))
        self.fps = float(c["fps"])
        self.contacts = c["contacts"]
        self.rallies = r["rallies"]

    def in_rally(self, frame: int) -> bool:
        t = frame / self.fps
        return any(ra["start_t"] <= t <= ra["end_t"] for ra in self.rallies)

    def banner(self, frame: int) -> str:
        t = frame / self.fps
        for ra in self.rallies:
            if ra["start_t"] - 0.5 <= t <= ra["end_t"] + 1.0:
                serve = ra["serving_team"] or "?"
                how = {-2: "traced", -3: "inferred", None: "not found"}.get(
                    ra["serve_contact"], "seen")
                state = "" if t <= ra["end_t"] else f" - ended ({ra['end_reason']})"
                return (f"RALLY {ra['index']}  serve: {serve} ({how})  "
                        f"conf {ra['confidence']:.2f}{state}")
        return "between rallies"

    def draw(self, image: np.ndarray, frame: int, s: float) -> str:
        """The banner text (touch markers are drawn by the ball overlay)."""
        return self.banner(frame)
