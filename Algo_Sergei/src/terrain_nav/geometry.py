from __future__ import annotations

import numpy as np


def compute_trajectory_points(
    start_x_m: float,
    start_y_m: float,
    speed_mps: float,
    azimuth_deg: float,
    timestamps_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth_rad = np.deg2rad(azimuth_deg)
    distances = speed_mps * (timestamps_s - timestamps_s[0])

    dx = distances * np.sin(azimuth_rad)
    dy = distances * np.cos(azimuth_rad)

    return start_x_m + dx, start_y_m + dy


def smooth_trajectory(
    x_m: np.ndarray,
    y_m: np.ndarray,
    window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1 or x_m.size < 3:
        return x_m.copy(), y_m.copy()

    window = min(int(window), x_m.size)
    if window % 2 == 0:
        window += 1
    if window > x_m.size:
        window = x_m.size if x_m.size % 2 == 1 else x_m.size - 1

    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2

    x_padded = np.pad(x_m, pad_width=pad, mode="edge")
    y_padded = np.pad(y_m, pad_width=pad, mode="edge")

    return (
        np.convolve(x_padded, kernel, mode="valid"),
        np.convolve(y_padded, kernel, mode="valid"),
    )
