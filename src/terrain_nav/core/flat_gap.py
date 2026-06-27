from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terrain_nav.core.geometry import compute_trajectory_points


@dataclass(frozen=True)
class FlatGap:
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    variance_m2: float


@dataclass(frozen=True)
class FlatGapBridge:
    gap: FlatGap
    bridge_x_m: np.ndarray
    bridge_y_m: np.ndarray
    confidence: np.ndarray
    predicted_exit_x_m: float
    predicted_exit_y_m: float
    correction_x_m: float
    correction_y_m: float


def detect_flat_gap(
    profile_m: np.ndarray,
    timestamps_s: np.ndarray,
    window_points: int = 31,
    variance_threshold_m2: float = 4.0,
    gradient_threshold_m: float = 1.0,
    min_duration_s: float = 20.0,
    min_start_index: int = 0,
) -> FlatGap | None:
    profile = np.asarray(profile_m, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if profile.size != timestamps.size:
        raise ValueError("profile_m and timestamps_s must have the same size")
    if profile.size < 3:
        return None

    window_points = max(3, int(window_points))
    if window_points % 2 == 0:
        window_points += 1
    half = window_points // 2

    fill = float(np.nanmedian(profile)) if np.isfinite(profile).any() else 0.0
    clean = np.nan_to_num(profile, nan=fill)
    gradient = np.abs(np.gradient(clean))
    local_variance = np.empty(clean.shape, dtype=np.float64)

    for index in range(clean.size):
        begin = max(0, index - half)
        end = min(clean.size, index + half + 1)
        local_variance[index] = float(np.var(clean[begin:end]))

    flat_mask = (local_variance <= variance_threshold_m2) & (gradient <= gradient_threshold_m)
    ranges = _true_ranges(flat_mask)
    if not ranges:
        return None

    best: FlatGap | None = None
    for start, end in ranges:
        if start < min_start_index:
            continue
        duration = float(timestamps[end] - timestamps[start])
        if duration < min_duration_s:
            continue
        variance = float(np.var(clean[start : end + 1]))
        if variance > variance_threshold_m2:
            continue
        gap = FlatGap(
            start_index=int(start),
            end_index=int(end),
            start_time_s=float(timestamps[start]),
            end_time_s=float(timestamps[end]),
            duration_s=duration,
            variance_m2=variance,
        )
        if best is None or gap.duration_s > best.duration_s:
            best = gap

    return best


def bridge_flat_gap(
    timestamps_s: np.ndarray,
    gap: FlatGap,
    entry_x_m: float,
    entry_y_m: float,
    speed_mps: float,
    azimuth_deg: float,
    exit_anchor_x_m: float | None = None,
    exit_anchor_y_m: float | None = None,
    entry_confidence: float = 1.0,
    decay_time_s: float = 180.0,
) -> FlatGapBridge:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if not 0 <= gap.start_index <= gap.end_index < timestamps.size:
        raise ValueError("gap indexes are outside timestamps")

    gap_timestamps = timestamps[gap.start_index : gap.end_index + 1]
    local_timestamps = gap_timestamps - gap_timestamps[0]
    bridge_x, bridge_y = compute_trajectory_points(
        start_x_m=float(entry_x_m),
        start_y_m=float(entry_y_m),
        speed_mps=float(speed_mps),
        azimuth_deg=float(azimuth_deg),
        timestamps_s=local_timestamps,
    )

    predicted_exit_x = float(bridge_x[-1])
    predicted_exit_y = float(bridge_y[-1])
    correction_x = 0.0
    correction_y = 0.0

    if exit_anchor_x_m is not None and exit_anchor_y_m is not None:
        correction_x = float(exit_anchor_x_m) - predicted_exit_x
        correction_y = float(exit_anchor_y_m) - predicted_exit_y
        alpha = np.linspace(0.0, 1.0, bridge_x.size, dtype=np.float64)
        bridge_x = bridge_x + correction_x * alpha
        bridge_y = bridge_y + correction_y * alpha

    decay_time_s = max(float(decay_time_s), 1e-6)
    confidence = float(np.clip(entry_confidence, 0.0, 1.0)) * np.exp(-local_timestamps / decay_time_s)

    return FlatGapBridge(
        gap=gap,
        bridge_x_m=bridge_x,
        bridge_y_m=bridge_y,
        confidence=confidence,
        predicted_exit_x_m=predicted_exit_x,
        predicted_exit_y_m=predicted_exit_y,
        correction_x_m=correction_x,
        correction_y_m=correction_y,
    )


def _true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            ranges.append((start, index - 1))
            start = None
    if start is not None:
        ranges.append((start, mask.size - 1))
    return ranges


def _close_false_gaps(mask: np.ndarray, max_gap_points: int) -> np.ndarray:
    closed = np.asarray(mask, dtype=bool).copy()
    false_ranges = _true_ranges(~closed)
    for start, end in false_ranges:
        if start == 0 or end == closed.size - 1:
            continue
        if end - start + 1 <= max_gap_points:
            closed[start : end + 1] = True
    return closed
