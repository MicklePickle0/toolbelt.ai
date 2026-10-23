"""Overlapping tile grid over a frame, shared by tiled training export and tiled inference.

The ball is ~20-60 px in 4K footage. Feeding a whole frame to a 1280 px model shrinks it
3x; a 3x3 grid of native-resolution tiles keeps it near full size.
"""

from __future__ import annotations

import math


def parse_grid(text: str) -> tuple[int, int]:
    """'3x3' -> (columns, rows)."""
    cols, rows = (int(v) for v in text.lower().split("x"))
    return cols, rows


def tile_boxes(width: int, height: int, grid: tuple[int, int],
               overlap: float) -> list[tuple[int, int, int, int]]:
    """Tiles as (x, y, w, h), row by row, covering the frame with `overlap` between
    neighbours (a share of the tile size)."""
    cols, rows = grid
    tw = min(width, math.ceil(width / (cols - overlap * (cols - 1))))
    th = min(height, math.ceil(height / (rows - overlap * (rows - 1))))
    return [(min(round(i * tw * (1 - overlap)), width - tw),
             min(round(j * th * (1 - overlap)), height - th), tw, th)
            for j in range(rows) for i in range(cols)]
