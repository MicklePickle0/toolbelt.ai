"""Draw rallies and contacts: a banner with the rally number, serving team and how it ended,
and a marker at each touch / serve / floor contact for a moment around it."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

COLOURS = {"us": (0, 220, 255), "them": (255, 120, 60), None: (200, 200, 200),
           "floor": (0, 0, 255)}


class RallyOverlay:
    def __init__(self, cache_dir: Path, show_s: float = 0.5):
        c = json.loads((cache_dir / "contacts.json").read_text(encoding="utf-8"))
        r = json.loads((cache_dir / "rallies.json").read_text(encoding="utf-8"))
        self.fps = float(c["fps"])
        self.contacts = c["contacts"]
        self.rallies = r["rallies"]
        self.show_s = show_s
        self.in_rally = {i for ra in self.rallies for i in ra["contacts"]}

    def banner(self, frame: int) -> str:
        t = frame / self.fps
        for ra in self.rallies:
            if ra["start_t"] - 0.5 <= t <= ra["end_t"] + 1.0:
                serve = ra["serving_team"] or "?"
                how = "traced" if ra["serve_contact"] == -2 else "seen" \
                    if ra["serve_contact"] is not None else "not found"
                state = "" if t <= ra["end_t"] else f" - ended ({ra['end_reason']})"
                return (f"RALLY {ra['index']}  serve: {serve} ({how})  "
                        f"conf {ra['confidence']:.2f}{state}")
        return "between rallies"

    def draw(self, image: np.ndarray, frame: int, s: float) -> str:
        t = frame / self.fps
        th = max(1, round(image.shape[1] / 960))
        for i, c in enumerate(self.contacts):
            if abs(c["t"] - t) > self.show_s:
                continue
            u, v = (np.array(c["ball_uv"]) * s).round().astype(int)
            colour = COLOURS["floor"] if c["kind"] == "floor" else COLOURS[c["team"]]
            faded = i not in self.in_rally
            r = 22 * th
            cv2.circle(image, (u, v), r, colour, th if faded else 3 * th, cv2.LINE_AA)
            label = "SERVE" if c["serve"] else c["kind"].upper()
            label += f" {c['team'] or ''} {c['confidence']:.2f}"
            if c["track_id"] >= 0 and c["kind"] == "touch":
                label += f" #{c['track_id']}"
            cv2.putText(image, label, (u + r + 4, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th,
                        (0, 0, 0), 4 * th, cv2.LINE_AA)
            cv2.putText(image, label, (u + r + 4, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th,
                        colour, th, cv2.LINE_AA)
        return self.banner(frame)
