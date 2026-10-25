"""3D court geometry (lines, net, keypoints) from config/court.yaml and the game's net height."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from setter_eyes import config

FLOOR_LINES = ("end_near", "attack_near", "centre", "attack_far", "end_far",
               "side_left", "side_right")
NET_LINES = ("net_top", "net_bottom", "antenna_left", "antenna_right")


@dataclass(frozen=True)
class CourtModel:
    net_height: float
    keypoints: dict[str, np.ndarray]
    lines: dict[str, np.ndarray]          # name -> (2, 3) endpoints of the painted line's centre

    @property
    def line_names(self) -> list[str]:
        return list(self.lines)

    def samples(self, step: float = 0.1,
                names: tuple[str, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Points every `step` metres along the lines -> (N, 3) points, (N,) line index."""
        pts, idx = [], []
        for i, name in enumerate(self.lines):
            if names is not None and name not in names:
                continue
            a, b = self.lines[name]
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
            pts.append(np.linspace(a, b, n))
            idx.append(np.full(n, i))
        return np.concatenate(pts), np.concatenate(idx)


def court_model(net_height_m: float, court: dict | None = None) -> CourtModel:
    court = court or config.load("court")
    half = court["length"] / 2
    w = court["width"]
    att = court["attack_line_distance"]
    hw = court["line_width"] / 2          # painted lines lie inside the court boundary
    net = court["net"]
    h = net_height_m
    tape = h - net["tape_width"] / 2
    bottom = h - net["width"]
    antenna_top = h + net["antenna_height_above_tape"]

    def seg(a, b):
        return np.array([a, b], float)

    lines = {
        "end_near": seg((0, half - hw, 0), (w, half - hw, 0)),
        "attack_near": seg((0, att - hw, 0), (w, att - hw, 0)),
        "centre": seg((0, 0, 0), (w, 0, 0)),
        "attack_far": seg((0, -att + hw, 0), (w, -att + hw, 0)),
        "end_far": seg((0, -half + hw, 0), (w, -half + hw, 0)),
        "side_left": seg((hw, -half, 0), (hw, half, 0)),
        "side_right": seg((w - hw, -half, 0), (w - hw, half, 0)),
        "net_top": seg((0, 0, tape), (w, 0, tape)),
        "net_bottom": seg((0, 0, bottom), (w, 0, bottom)),
        "antenna_left": seg((0, 0, bottom), (0, 0, antenna_top)),
        "antenna_right": seg((w, 0, bottom), (w, 0, antenna_top)),
    }
    keypoints = {
        name: np.array([x, y, h if z is None else z], float)
        for name, (x, y, z) in court["keypoints"].items()
    }
    return CourtModel(net_height=h, keypoints=keypoints, lines=lines)
