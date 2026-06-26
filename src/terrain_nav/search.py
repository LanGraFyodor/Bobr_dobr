from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS

from terrain_nav.dem import dataset_center_lonlat, dataset_center_utm, dataset_utm_bounds, utm_crs_for_lonlat
from terrain_nav.geometry import compute_trajectory_points, smooth_trajectory
from terrain_nav.models import LocalizationResult, NmeaProfile, SearchGrid
from terrain_nav.nmea import parse_nmea_profile
from terrain_nav.sampling import sample_dem_heights
from terrain_nav.matching import (
    compute_error_grid,
    make_local_azimuths,
    make_local_values,
    make_refine_start_points,
    make_start_points,
    refine_best_candidates,
    search_start_points,
)


def make_search_grid(
    min_speed_mps: float = 10.0,
    max_speed_mps: float = 50.0,
    speed_step_mps: float = 1.0,
    azimuth_step_deg: float = 1.0,
) -> SearchGrid:
    if speed_step_mps <= 0:
        raise ValueError("speed_step_mps must be positive")
    if azimuth_step_deg <= 0:
        raise ValueError("azimuth_step_deg must be positive")

    speeds = np.arange(min_speed_mps, max_speed_mps + speed_step_mps * 0.5, speed_step_mps)
    azimuths = np.arange(0.0, 360.0, azimuth_step_deg)
    return SearchGrid(speeds_mps=speeds, azimuths_deg=azimuths)


def localize_from_nmea(
    dem_path: str | Path,
    nmea_path: str | Path,
    start_x_m: float | None = None,
    start_y_m: float | None = None,
    previous_x_m: float | None = None,
    previous_y_m: float | None = None,
    baro_altitude_m: float = 1500.0,
    sample_rate_hz: float = 1.0,
    min_speed_mps: float = 10.0,
    max_speed_mps: float = 50.0,
    speed_step_mps: float = 1.0,
    azimuth_step_deg: float = 1.0,
    batch_size: int = 512,
    flat_variance_threshold_m2: float = 1.0,
    smoothing_window: int = 5,
    max_profile_points: int | None = 600,
) -> LocalizationResult:
    profile = parse_nmea_profile(nmea_path, baro_altitude_m, sample_rate_hz)
    profile = _limit_profile_points(profile, max_profile_points)
    grid = make_search_grid(min_speed_mps, max_speed_mps, speed_step_mps, azimuth_step_deg)

    with rasterio.open(dem_path) as dataset:
        dem, source_crs, utm_crs = _load_dem_context(dataset)
        if start_x_m is None or start_y_m is None:
            start_x_m, start_y_m = dataset_center_utm(dataset, source_crs, utm_crs)

        terrain_variance = float(np.nanvar(profile.terrain_profile_m))
        if terrain_variance <= flat_variance_threshold_m2:
            old_x = float(start_x_m if previous_x_m is None else previous_x_m)
            old_y = float(start_y_m if previous_y_m is None else previous_y_m)
            return _flat_result(profile.terrain_profile_m, profile.timestamps_s, grid, old_x, old_y, terrain_variance)

        errors, best_flat_index = compute_error_grid(
            dem=dem,
            dataset=dataset,
            source_crs=source_crs,
            utm_crs=utm_crs,
            measured_profile_m=profile.terrain_profile_m,
            timestamps_s=profile.timestamps_s,
            start_x_m=float(start_x_m),
            start_y_m=float(start_y_m),
            speeds_mps=grid.speeds_mps,
            azimuths_deg=grid.azimuths_deg,
        )

        best_speed_index, best_azimuth_index = np.unravel_index(best_flat_index, errors.shape)
        result = _build_result(
            dem=dem,
            dataset=dataset,
            source_crs=source_crs,
            utm_crs=utm_crs,
            measured_profile_m=profile.terrain_profile_m,
            timestamps_s=profile.timestamps_s,
            speeds_mps=grid.speeds_mps,
            azimuths_deg=grid.azimuths_deg,
            start_x_m=float(start_x_m),
            start_y_m=float(start_y_m),
            speed_mps=float(grid.speeds_mps[best_speed_index]),
            azimuth_deg=float(grid.azimuths_deg[best_azimuth_index]),
            best_error=float(errors[best_speed_index, best_azimuth_index]),
            best_speed_index=int(best_speed_index),
            best_azimuth_index=int(best_azimuth_index),
            errors=errors,
            terrain_variance_m2=terrain_variance,
            smoothing_window=smoothing_window,
        )

    return result


def localize_position_from_nmea(
    dem_path: str | Path,
    nmea_path: str | Path,
    previous_x_m: float | None = None,
    previous_y_m: float | None = None,
    baro_altitude_m: float = 1500.0,
    sample_rate_hz: float = 1.0,
    min_speed_mps: float = 10.0,
    max_speed_mps: float = 50.0,
    coarse_speed_step_mps: float = 5.0,
    fine_speed_step_mps: float = 1.0,
    coarse_azimuth_step_deg: float = 10.0,
    fine_azimuth_step_deg: float = 1.0,
    coarse_start_step_m: float = 10_000.0,
    refine_radius_m: float = 5_000.0,
    refine_start_step_m: float = 1_000.0,
    coarse_top_k: int = 10,
    flat_variance_threshold_m2: float = 1.0,
    smoothing_window: int = 5,
    max_profile_points: int | None = 600,
) -> LocalizationResult:
    profile = parse_nmea_profile(nmea_path, baro_altitude_m, sample_rate_hz)
    profile = _limit_profile_points(profile, max_profile_points)

    with rasterio.open(dem_path) as dataset:
        dem, source_crs, utm_crs = _load_dem_context(dataset)
        center_x_m, center_y_m = dataset_center_utm(dataset, source_crs, utm_crs)

        coarse_grid = make_search_grid(
            min_speed_mps=min_speed_mps,
            max_speed_mps=max_speed_mps,
            speed_step_mps=coarse_speed_step_mps,
            azimuth_step_deg=coarse_azimuth_step_deg,
        )

        terrain_variance = float(np.nanvar(profile.terrain_profile_m))
        if terrain_variance <= flat_variance_threshold_m2:
            old_x = float(center_x_m if previous_x_m is None else previous_x_m)
            old_y = float(center_y_m if previous_y_m is None else previous_y_m)
            return _flat_result(profile.terrain_profile_m, profile.timestamps_s, coarse_grid, old_x, old_y, terrain_variance)

        bounds = dataset_utm_bounds(dataset, source_crs, utm_crs)
        coarse_starts = make_start_points(
            bounds=bounds,
            step_m=coarse_start_step_m,
            extra_points=np.array([[center_x_m, center_y_m]], dtype=np.float64),
        )
        coarse = search_start_points(
            dem=dem,
            dataset=dataset,
            source_crs=source_crs,
            utm_crs=utm_crs,
            measured_profile_m=profile.terrain_profile_m,
            timestamps_s=profile.timestamps_s,
            start_points=coarse_starts,
            speeds_mps=coarse_grid.speeds_mps,
            azimuths_deg=coarse_grid.azimuths_deg,
                top_k=coarse_top_k,
        )

        fine, fine_speeds, fine_azimuths = refine_best_candidates(
            coarse_candidates=coarse["candidates"],
            bounds=bounds,
            dem=dem,
            dataset=dataset,
            source_crs=source_crs,
            utm_crs=utm_crs,
            measured_profile_m=profile.terrain_profile_m,
            timestamps_s=profile.timestamps_s,
            min_speed_mps=min_speed_mps,
            max_speed_mps=max_speed_mps,
            coarse_speed_step_mps=coarse_speed_step_mps,
            fine_speed_step_mps=fine_speed_step_mps,
            coarse_azimuth_step_deg=coarse_azimuth_step_deg,
            fine_azimuth_step_deg=fine_azimuth_step_deg,
            refine_radius_m=refine_radius_m,
            refine_start_step_m=refine_start_step_m,
        )

        result = _build_result(
            dem=dem,
            dataset=dataset,
            source_crs=source_crs,
            utm_crs=utm_crs,
            measured_profile_m=profile.terrain_profile_m,
            timestamps_s=profile.timestamps_s,
            speeds_mps=fine_speeds,
            azimuths_deg=fine_azimuths,
            start_x_m=float(fine["start_x"]),
            start_y_m=float(fine["start_y"]),
            speed_mps=float(fine["speed"]),
            azimuth_deg=float(fine["azimuth"]),
            best_error=float(fine["error"]),
            best_speed_index=int(fine["speed_index"]),
            best_azimuth_index=int(fine["azimuth_index"]),
            errors=fine["errors"],
            terrain_variance_m2=terrain_variance,
            smoothing_window=smoothing_window,
        )

    return result


def compute_confidence(errors: np.ndarray) -> float:
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return 0.0

    best = float(np.min(finite))
    median = float(np.median(finite))
    p10 = float(np.percentile(finite, 10.0))
    eps = 1e-9

    global_contrast = (median - best) / (median + eps)
    spot_contrast = (p10 - best) / (p10 + eps)
    confidence = np.sqrt(max(global_contrast, 0.0) * max(spot_contrast, 0.0))
    return float(np.clip(confidence, 0.0, 1.0))


def _limit_profile_points(profile: NmeaProfile, max_points: int | None) -> NmeaProfile:
    if max_points is None or max_points <= 0 or profile.timestamps_s.size <= max_points:
        return profile

    indexes = np.unique(np.linspace(0, profile.timestamps_s.size - 1, int(max_points), dtype=int))
    return NmeaProfile(
        radio_altitudes_m=profile.radio_altitudes_m[indexes],
        terrain_profile_m=profile.terrain_profile_m[indexes],
        timestamps_s=profile.timestamps_s[indexes],
    )


def _load_dem_context(dataset: rasterio.io.DatasetReader) -> tuple[np.ndarray, CRS, CRS]:
    dem = dataset.read(1).astype(np.float64)
    if dataset.nodata is not None:
        dem[dem == dataset.nodata] = np.nan

    source_crs = CRS.from_user_input(dataset.crs)
    center_lon, center_lat = dataset_center_lonlat(dataset, source_crs)
    utm_crs = utm_crs_for_lonlat(center_lon, center_lat)
    return dem, source_crs, utm_crs


def _flat_result(
    measured_profile_m: np.ndarray,
    timestamps_s: np.ndarray,
    grid: SearchGrid,
    x_m: float,
    y_m: float,
    terrain_variance_m2: float,
) -> LocalizationResult:
    fallback_x = np.full(timestamps_s.shape, x_m, dtype=np.float64)
    fallback_y = np.full(timestamps_s.shape, y_m, dtype=np.float64)
    errors = np.full((grid.speeds_mps.size, grid.azimuths_deg.size), np.nan)

    return LocalizationResult(
        speed_mps=0.0,
        azimuth_deg=0.0,
        start_x_m=x_m,
        start_y_m=y_m,
        current_x_m=x_m,
        current_y_m=y_m,
        best_error=float("nan"),
        best_speed_index=-1,
        best_azimuth_index=-1,
        errors=errors,
        speeds_mps=grid.speeds_mps,
        azimuths_deg=grid.azimuths_deg,
        trajectory_x_m=fallback_x,
        trajectory_y_m=fallback_y,
        measured_profile_m=measured_profile_m,
        predicted_profile_m=np.full(measured_profile_m.shape, np.nan),
        smoothed_trajectory_x_m=fallback_x,
        smoothed_trajectory_y_m=fallback_y,
        terrain_variance_m2=terrain_variance_m2,
        confidence=0.0,
        is_flat_terrain=True,
    )


def _build_result(
    dem: np.ndarray,
    dataset: rasterio.io.DatasetReader,
    source_crs: CRS,
    utm_crs: CRS,
    measured_profile_m: np.ndarray,
    timestamps_s: np.ndarray,
    speeds_mps: np.ndarray,
    azimuths_deg: np.ndarray,
    start_x_m: float,
    start_y_m: float,
    speed_mps: float,
    azimuth_deg: float,
    best_error: float,
    best_speed_index: int,
    best_azimuth_index: int,
    errors: np.ndarray,
    terrain_variance_m2: float,
    smoothing_window: int,
) -> LocalizationResult:
    trajectory_x, trajectory_y = compute_trajectory_points(start_x_m, start_y_m, speed_mps, azimuth_deg, timestamps_s)
    predicted = sample_dem_heights(dem, dataset, source_crs, utm_crs, trajectory_x, trajectory_y)
    smoothed_x, smoothed_y = smooth_trajectory(trajectory_x, trajectory_y, window=smoothing_window)

    return LocalizationResult(
        speed_mps=speed_mps,
        azimuth_deg=azimuth_deg,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        current_x_m=float(trajectory_x[-1]),
        current_y_m=float(trajectory_y[-1]),
        best_error=best_error,
        best_speed_index=best_speed_index,
        best_azimuth_index=best_azimuth_index,
        errors=errors,
        speeds_mps=speeds_mps,
        azimuths_deg=azimuths_deg,
        trajectory_x_m=trajectory_x,
        trajectory_y_m=trajectory_y,
        measured_profile_m=measured_profile_m,
        predicted_profile_m=predicted,
        smoothed_trajectory_x_m=smoothed_x,
        smoothed_trajectory_y_m=smoothed_y,
        terrain_variance_m2=terrain_variance_m2,
        confidence=compute_confidence(errors),
        is_flat_terrain=False,
    )
