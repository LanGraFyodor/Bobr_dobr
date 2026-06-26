from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS


@dataclass(frozen=True)
class DemData:
    heights: np.ndarray
    x_utm: np.ndarray
    y_utm: np.ndarray
    source_crs: CRS
    utm_crs: CRS
    nodata: float | int | None


@dataclass(frozen=True)
class NmeaProfile:
    radio_altitudes_m: np.ndarray
    terrain_profile_m: np.ndarray
    timestamps_s: np.ndarray


@dataclass(frozen=True)
class SearchGrid:
    speeds_mps: np.ndarray
    azimuths_deg: np.ndarray


@dataclass(frozen=True)
class LocalizationResult:
    speed_mps: float
    azimuth_deg: float
    start_x_m: float
    start_y_m: float
    current_x_m: float
    current_y_m: float
    best_error: float
    best_speed_index: int
    best_azimuth_index: int
    errors: np.ndarray
    speeds_mps: np.ndarray
    azimuths_deg: np.ndarray
    trajectory_x_m: np.ndarray
    trajectory_y_m: np.ndarray
    measured_profile_m: np.ndarray
    predicted_profile_m: np.ndarray
    smoothed_trajectory_x_m: np.ndarray
    smoothed_trajectory_y_m: np.ndarray
    terrain_variance_m2: float
    confidence: float
    is_flat_terrain: bool


@dataclass(frozen=True)
class GeneratedFlight:
    timestamps_s: np.ndarray
    trajectory_x_m: np.ndarray
    trajectory_y_m: np.ndarray
    terrain_heights_m: np.ndarray
    radio_altitudes_m: np.ndarray
    noisy_radio_altitudes_m: np.ndarray
    nmea_lines: list[str]
