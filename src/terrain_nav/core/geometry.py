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


def kalman_smooth_trajectory(
    x_m: np.ndarray,
    y_m: np.ndarray,
    timestamps_s: np.ndarray,
    process_noise: float = 1.0,
    measurement_noise: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    if x_m.size < 3:
        return x_m.copy(), y_m.copy()

    state = np.array([x_m[0], y_m[0], 0.0, 0.0], dtype=np.float64)
    covariance = np.diag([measurement_noise, measurement_noise, 100.0, 100.0]).astype(np.float64)
    observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    observation_noise = np.eye(2, dtype=np.float64) * measurement_noise

    smoothed_x = np.empty_like(x_m, dtype=np.float64)
    smoothed_y = np.empty_like(y_m, dtype=np.float64)
    smoothed_x[0] = x_m[0]
    smoothed_y[0] = y_m[0]

    for index in range(1, x_m.size):
        dt = max(float(timestamps_s[index] - timestamps_s[index - 1]), 1e-3)
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        q = max(float(process_noise), 1e-9)
        process = q * np.array(
            [
                [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
                [dt**3 / 2.0, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2.0, 0.0, dt**2],
            ],
            dtype=np.float64,
        )

        state = transition @ state
        covariance = transition @ covariance @ transition.T + process

        measurement = np.array([x_m[index], y_m[index]], dtype=np.float64)
        innovation = measurement - observation @ state
        innovation_covariance = observation @ covariance @ observation.T + observation_noise
        gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        state = state + gain @ innovation
        covariance = (np.eye(4, dtype=np.float64) - gain @ observation) @ covariance

        smoothed_x[index] = state[0]
        smoothed_y[index] = state[1]

    return smoothed_x, smoothed_y
