"""Manual-click fallback: click known court points on a frame (OpenCV window).

Keys: click = place the prompted point, s = skip it (not visible), u = undo,
i/j/k/l = nudge the last click 1 px (up/left/down/right, full resolution),
Enter = done, q/Esc = cancel. A magnified view follows the mouse.
"""

from __future__ import annotations

import cv2
import numpy as np

from setter_eyes.calibration.court_model import CourtModel

WINDOW = "calibrate"
NUDGE = {ord("i"): (0, -1), ord("j"): (-1, 0), ord("k"): (0, 1), ord("l"): (1, 0)}


def _magnifier(image: np.ndarray, centre: tuple[float, float], radius: int = 60,
               zoom: int = 4) -> np.ndarray:
    h, w = image.shape[:2]
    x, y = round(centre[0]), round(centre[1])
    pad = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
    crop = pad[y:y + 2 * radius + 1, x:x + 2 * radius + 1]
    if crop.shape[0] != 2 * radius + 1 or crop.shape[1] != 2 * radius + 1 or x >= w or y >= h:
        crop = np.zeros((2 * radius + 1, 2 * radius + 1, 3), np.uint8)
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    c = big.shape[0] // 2
    cv2.line(big, (c, 0), (c, big.shape[0]), (0, 0, 255), 1)
    cv2.line(big, (0, c), (big.shape[1], c), (0, 0, 255), 1)
    return big


def click_keypoints(image: np.ndarray, model: CourtModel,
                    max_display: tuple[int, int] = (1600, 900)) -> dict[str, list[float]] | None:
    """-> {keypoint name: [u, v]} in full-resolution pixels, or None if cancelled."""
    names = list(model.keypoints)
    h, w = image.shape[:2]
    s = min(1.0, max_display[0] / w, max_display[1] / h)
    display = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    state = {"i": 0, "mouse": (w / 2, h / 2), "order": []}
    clicks: dict[str, list[float]] = {}

    def on_mouse(event, x, y, _flags, _param):
        state["mouse"] = (x / s, y / s)
        if event == cv2.EVENT_LBUTTONDOWN and state["i"] < len(names):
            name = names[state["i"]]
            clicks[name] = [x / s, y / s]
            state["order"].append(name)
            state["i"] += 1

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            frame = display.copy()
            for name, (u, v) in clicks.items():
                p = (round(u * s), round(v * s))
                cv2.circle(frame, p, 4, (0, 128, 255), -1)
                cv2.putText(frame, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 128, 255), 1, cv2.LINE_AA)
            prompt = (f"[{state['i'] + 1}/{len(names)}] click {names[state['i']]}"
                      if state["i"] < len(names) else "all points done")
            prompt += "   s=skip u=undo ijkl=nudge Enter=done q=cancel"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(frame, prompt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                        1, cv2.LINE_AA)
            mag = _magnifier(image, state["mouse"])
            frame[30:30 + mag.shape[0], -mag.shape[1] - 4:-4] = mag[: frame.shape[0] - 30]
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key in (13, 10):
                return clicks
            if key == ord("s") and state["i"] < len(names):
                state["i"] += 1
            elif key == ord("u") and state["i"] > 0:
                state["i"] -= 1
                clicks.pop(names[state["i"]], None)
            elif key in NUDGE and state["order"]:
                last = state["order"][-1]
                if last in clicks:
                    dx, dy = NUDGE[key]
                    clicks[last] = [clicks[last][0] + dx, clicks[last][1] + dy]
    finally:
        cv2.destroyWindow(WINDOW)


# Lines that can be clicked anywhere along their visible part, in prompt order.
CLICKABLE_LINES = ("end_near", "side_left", "side_right", "attack_near", "centre",
                   "attack_far", "end_far", "net_top")
LINE_HINTS = {"net_top": "middle of the white top tape"}


def click_lines(image: np.ndarray, keypoints: dict[str, list[float]],
                lines: tuple[str, ...] = CLICKABLE_LINES,
                max_display: tuple[int, int] = (1600, 900)) -> dict[str, list[list[float]]] | None:
    """Click any number of points along each court line (on the middle of the painted
    line); useful where its corners are hidden -> {line name: [[u, v], ...]}, or None.

    Keys: click = add a point on the prompted line, n = next line (or skip it),
    u = undo the last point (back to the previous line if this one has none),
    ijkl = nudge the last point 1 px, Enter = done, q/Esc = cancel.
    """
    h, w = image.shape[:2]
    s = min(1.0, max_display[0] / w, max_display[1] / h)
    display = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    state = {"i": 0, "mouse": (w / 2, h / 2)}
    clicks: dict[str, list[list[float]]] = {name: [] for name in lines}

    def on_mouse(event, x, y, _flags, _param):
        state["mouse"] = (x / s, y / s)
        if event == cv2.EVENT_LBUTTONDOWN and state["i"] < len(lines):
            clicks[lines[state["i"]]].append([x / s, y / s])

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            frame = display.copy()
            for u, v in keypoints.values():
                cv2.circle(frame, (round(u * s), round(v * s)), 4, (0, 128, 255), -1)
            for j, name in enumerate(lines):
                colour = (0, 255, 255) if j == state["i"] else (255, 200, 0)
                for u, v in clicks[name]:
                    cv2.drawMarker(frame, (round(u * s), round(v * s)), colour,
                                   cv2.MARKER_TILTED_CROSS, 10, 2)
            if state["i"] < len(lines):
                name = lines[state["i"]]
                where = LINE_HINTS.get(name, "middle of the painted line")
                prompt = (f"[{state['i'] + 1}/{len(lines)}] click points along {name} ({where}),"
                          f" {len(clicks[name])} so far")
            else:
                prompt = "all lines done"
            prompt += "   n=next u=undo ijkl=nudge Enter=done q=cancel"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(frame, prompt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1, cv2.LINE_AA)
            mag = _magnifier(image, state["mouse"])
            frame[30:30 + mag.shape[0], -mag.shape[1] - 4:-4] = mag[: frame.shape[0] - 30]
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key in (13, 10):
                break
            current = clicks[lines[state["i"]]] if state["i"] < len(lines) else None
            if key in (ord("n"), ord("s")) and state["i"] < len(lines):
                state["i"] += 1
            elif key == ord("u"):
                if current:
                    current.pop()
                elif state["i"] > 0:
                    state["i"] -= 1
            elif key in NUDGE and current:
                dx, dy = NUDGE[key]
                current[-1] = [current[-1][0] + dx, current[-1][1] + dy]
    finally:
        cv2.destroyWindow(WINDOW)
    return {name: pts for name, pts in clicks.items() if pts}


def confirm(image: np.ndarray, max_display: tuple[int, int] = (1600, 900)) -> bool:
    """Show an overlay; y = accept, anything else = reject."""
    h, w = image.shape[:2]
    s = min(1.0, max_display[0] / w, max_display[1] / h)
    frame = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.putText(frame, "y = accept, any other key = redo", (8, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(WINDOW)
    return key == ord("y")
