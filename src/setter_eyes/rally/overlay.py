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
        for n, ra in enumerate(self.rallies):
            if ra["start_t"] - 0.5 <= t <= ra["end_t"] + 2.5:
                serve = ra["serving_team"] or "?"
                sc = ra["serve_contact"]
                how = {-2: "traced", -3: "inferred", None: "not found"}.get(sc, "seen")
                if sc is not None and sc >= 0 and self.contacts[sc].get("entry_serve"):
                    how = "ball entered the frame"
                prev = self.rallies[n - 1] if n else {}
                score = f"us {prev.get('score_us', 0)} - {prev.get('score_them', 0)} them"
                if t <= ra["end_t"]:
                    return (f"RALLY {ra['index']}  {score}  serve: {serve} ({how})  "
                            f"conf {ra['confidence']:.2f}")
                winner = ra.get("point_winner")
                if winner is None:
                    return f"RALLY {ra['index']} ended ({ra['end_reason']}) - winner unknown"
                outcome = (ra.get("outcome") or "-").replace("_", " ").upper()
                return (f"POINT {winner.upper()} (they serve next, conf "
                        f"{ra.get('winner_confidence', 0):.2f}): {outcome}  ->  us "
                        f"{ra.get('score_us', 0)} - {ra.get('score_them', 0)} them")
        return "between rallies"

    def draw(self, image: np.ndarray, frame: int, s: float) -> str:
        """The banner text (touch markers are drawn by the ball overlay)."""
        return self.banner(frame)
