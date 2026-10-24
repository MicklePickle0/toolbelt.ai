"""Hand-written constant-acceleration Kalman filter for the ball in pan-free image pixels.

State [u, v, du, dv, ddu, ddv] (px, px/s, px/s^2). In the image a ballistic flight is
close to constant acceleration over short spans (gravity, foreshortened by depth), so
the acceleration is part of the state and starts at gravity's image-space pull; white
jerk noise lets it drift with perspective. Touches are velocity jumps the filter
absorbs over a few samples; the flight segmentation (arcs.py) finds them properly.
"""

from __future__ import annotations

import numpy as np

H = np.zeros((2, 6))
H[0, 0] = H[1, 1] = 1.0


def transition(dt: float) -> np.ndarray:
    F = np.eye(6)
    for i in range(2):
        F[i, 2 + i] = dt
        F[i, 4 + i] = dt * dt / 2
        F[2 + i, 4 + i] = dt
    return F


def process_noise(dt: float, jerk: float) -> np.ndarray:
    """White-jerk noise for one axis pair; jerk in px/s^3 (spectral density jerk**2)."""
    q = np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                  [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                  [dt**3 / 6, dt**2 / 2, dt]]) * jerk**2
    Q = np.zeros((6, 6))
    for i in range(2):
        idx = [i, 2 + i, 4 + i]
        Q[np.ix_(idx, idx)] = q
    return Q


class BallKalman:
    def __init__(self, uv: np.ndarray, t: float, cfg: dict, gravity_px: float):
        self.x = np.array([uv[0], uv[1], 0.0, 0.0, 0.0, gravity_px])
        p = cfg["init_sigma"]
        self.P = np.diag([p["pos"] ** 2] * 2 + [p["vel"] ** 2] * 2 + [p["acc"] ** 2] * 2)
        self.t = t
        self.jerk = cfg["jerk_px_s3"]

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        F = transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + process_noise(dt, self.jerk)
        self.t = t

    def peek(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Predicted (position, position covariance) at t without changing the state."""
        dt = max(0.0, t - self.t)
        F = transition(dt)
        x = F @ self.x
        P = F @ self.P @ F.T + process_noise(dt, self.jerk)
        return x[:2], P[:2, :2]

    def update(self, uv: np.ndarray, sigma_px: float) -> None:
        R = np.eye(2) * sigma_px**2
        y = uv - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
