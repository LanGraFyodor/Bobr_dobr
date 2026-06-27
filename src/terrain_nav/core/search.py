from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS

from terrain_nav.io.dem import dataset_center_lonlat, dataset_center_utm, dataset_utm_bounds, utm_crs_for_lonlat
from terrain_nav.core.geometry import compute_trajectory_points, kalman_smooth_trajectory, smooth_trajectory
from terrain_nav.models import LocalizationResult, NmeaProfile, SearchGrid
from terrain_nav.io.nmea import parse_nmea_profile
from terrain_nav.core.sampling import sample_dem_heights, sample_utm_raster_heights
from terrain_nav.core.matching import (
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
    smoothing_method: str = "kalman",
    use_weighted_scoring: bool = False,
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
            use_weighted_scoring=use_weighted_scoring,
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
            best_correlation=float("nan"),
            best_speed_index=int(best_speed_index),
            best_azimuth_index=int(best_azimuth_index),
            errors=errors,
            correlations=np.full_like(errors, np.nan, dtype=np.float64),
            terrain_variance_m2=terrain_variance,
            smoothing_window=smoothing_window,
            smoothing_method=smoothing_method,
            scoring_mode="weighted" if use_weighted_scoring else "regular",
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
    coarse_start_step_m: float = 5_000.0,
    refine_radius_m: float = 10_000.0,
    refine_start_step_m: float = 1_000.0,
    coarse_top_k: int = 50,
    flat_variance_threshold_m2: float = 1.0,
    smoothing_window: int = 5,
    smoothing_method: str = "kalman",
    use_weighted_scoring: bool = False,
    coarse_profile_points: int | None = 250,
    max_profile_points: int | None = 600,
    search_radius_m: float | None = None,
    use_rust_core: bool = True,
    allow_python_fallback: bool = True,
    auto_retry_unweighted: bool = True,
    auto_dense_retry: bool = True,
) -> LocalizationResult:
    profile = parse_nmea_profile(nmea_path, baro_altitude_m, sample_rate_hz)
    profile = _limit_profile_points(profile, max_profile_points)
    coarse_profile = _select_informative_profile_points(profile, coarse_profile_points)

    if use_rust_core:
        try:
            result = _localize_position_from_nmea_rust(
                dem_path=dem_path,
                profile=profile,
                previous_x_m=previous_x_m,
                previous_y_m=previous_y_m,
                min_speed_mps=min_speed_mps,
                max_speed_mps=max_speed_mps,
                coarse_speed_step_mps=coarse_speed_step_mps,
                fine_speed_step_mps=fine_speed_step_mps,
                coarse_azimuth_step_deg=coarse_azimuth_step_deg,
                fine_azimuth_step_deg=fine_azimuth_step_deg,
                coarse_start_step_m=coarse_start_step_m,
                refine_radius_m=refine_radius_m,
                refine_start_step_m=refine_start_step_m,
                coarse_top_k=coarse_top_k,
                flat_variance_threshold_m2=flat_variance_threshold_m2,
                smoothing_window=smoothing_window,
                smoothing_method=smoothing_method,
                use_weighted_scoring=use_weighted_scoring,
                coarse_profile=coarse_profile,
                search_radius_m=search_radius_m,
            )
            if auto_dense_retry and _needs_dense_retry(result, coarse_start_step_m, coarse_top_k):
                dense_result = localize_position_from_nmea(
                    dem_path=dem_path,
                    nmea_path=nmea_path,
                    previous_x_m=previous_x_m,
                    previous_y_m=previous_y_m,
                    baro_altitude_m=baro_altitude_m,
                    sample_rate_hz=sample_rate_hz,
                    min_speed_mps=min_speed_mps,
                    max_speed_mps=max_speed_mps,
                    coarse_speed_step_mps=coarse_speed_step_mps,
                    fine_speed_step_mps=fine_speed_step_mps,
                    coarse_azimuth_step_deg=coarse_azimuth_step_deg,
                    fine_azimuth_step_deg=fine_azimuth_step_deg,
                    coarse_start_step_m=1_000.0,
                    refine_radius_m=max(refine_radius_m, 10_000.0),
                    refine_start_step_m=min(refine_start_step_m, 500.0),
                    coarse_top_k=max(int(coarse_top_k), 200),
                    flat_variance_threshold_m2=flat_variance_threshold_m2,
                    smoothing_window=smoothing_window,
                    smoothing_method=smoothing_method,
                    use_weighted_scoring=use_weighted_scoring,
                    coarse_profile_points=coarse_profile_points,
                    max_profile_points=max_profile_points,
                    search_radius_m=search_radius_m,
                    use_rust_core=use_rust_core,
                    allow_python_fallback=allow_python_fallback,
                    auto_retry_unweighted=False,
                    auto_dense_retry=False,
                )
                if _is_better_result(dense_result, result):
                    result = _replace_scoring_mode(
                        dense_result,
                        f"{dense_result.scoring_mode}_dense_retry",
                    )
            if use_weighted_scoring and auto_retry_unweighted:
                regular_result = localize_position_from_nmea(
                    dem_path=dem_path,
                    nmea_path=nmea_path,
                    previous_x_m=previous_x_m,
                    previous_y_m=previous_y_m,
                    baro_altitude_m=baro_altitude_m,
                    sample_rate_hz=sample_rate_hz,
                    min_speed_mps=min_speed_mps,
                    max_speed_mps=max_speed_mps,
                    coarse_speed_step_mps=coarse_speed_step_mps,
                    fine_speed_step_mps=fine_speed_step_mps,
                    coarse_azimuth_step_deg=coarse_azimuth_step_deg,
                    fine_azimuth_step_deg=fine_azimuth_step_deg,
                    coarse_start_step_m=coarse_start_step_m,
                    refine_radius_m=refine_radius_m,
                    refine_start_step_m=refine_start_step_m,
                    coarse_top_k=coarse_top_k,
                    flat_variance_threshold_m2=flat_variance_threshold_m2,
                    smoothing_window=smoothing_window,
                    smoothing_method=smoothing_method,
                    use_weighted_scoring=False,
                    coarse_profile_points=coarse_profile_points,
                    max_profile_points=max_profile_points,
                    search_radius_m=search_radius_m,
                    use_rust_core=use_rust_core,
                    allow_python_fallback=allow_python_fallback,
                    auto_retry_unweighted=False,
                    auto_dense_retry=auto_dense_retry,
                )
                if _is_better_result(regular_result, result):
                    return _replace_scoring_mode(regular_result, "auto_fallback_regular")
            return result
        except Exception:
            if not allow_python_fallback:
                raise

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

        bounds = _clip_bounds_around_center(
            dataset_utm_bounds(dataset, source_crs, utm_crs),
            (center_x_m, center_y_m),
            search_radius_m,
        )
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
            measured_profile_m=coarse_profile.terrain_profile_m,
            timestamps_s=coarse_profile.timestamps_s,
            start_points=coarse_starts,
            speeds_mps=coarse_grid.speeds_mps,
            azimuths_deg=coarse_grid.azimuths_deg,
            top_k=coarse_top_k,
            use_weighted_scoring=use_weighted_scoring,
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
            use_weighted_scoring=use_weighted_scoring,
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
            best_correlation=float(fine.get("correlation", float("nan"))),
            best_speed_index=int(fine["speed_index"]),
            best_azimuth_index=int(fine["azimuth_index"]),
            errors=fine["errors"],
            correlations=np.full_like(fine["errors"], np.nan, dtype=np.float64),
            terrain_variance_m2=terrain_variance,
            smoothing_window=smoothing_window,
            smoothing_method=smoothing_method,
            scoring_mode="weighted" if use_weighted_scoring else "regular",
        )

    return result


def _localize_position_from_nmea_rust(
    dem_path: str | Path,
    profile: NmeaProfile,
    previous_x_m: float | None,
    previous_y_m: float | None,
    min_speed_mps: float,
    max_speed_mps: float,
    coarse_speed_step_mps: float,
    fine_speed_step_mps: float,
    coarse_azimuth_step_deg: float,
    fine_azimuth_step_deg: float,
    coarse_start_step_m: float,
    refine_radius_m: float,
    refine_start_step_m: float,
    coarse_top_k: int,
    flat_variance_threshold_m2: float,
    smoothing_window: int,
    smoothing_method: str,
    use_weighted_scoring: bool,
    coarse_profile: NmeaProfile,
    search_radius_m: float | None,
) -> LocalizationResult:
    from terrain_nav.core.rust_core import compute_error_grid_rust, search_candidates_rust
    from terrain_nav.core.utm_raster import load_utm_raster

    with rasterio.open(dem_path) as dataset:
        raster = load_utm_raster(dataset)

    center_x_m, center_y_m = raster.center
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

    search_bounds = _clip_bounds_around_center(raster.bounds, raster.center, search_radius_m)
    coarse_starts = make_start_points(
        bounds=search_bounds,
        step_m=coarse_start_step_m,
        extra_points=np.array([[center_x_m, center_y_m]], dtype=np.float64),
    )
    coarse_candidates = search_candidates_rust(
        dem=raster.heights,
        transform=raster.transform,
        measured_profile_m=coarse_profile.terrain_profile_m,
        timestamps_s=coarse_profile.timestamps_s,
        start_points_m=coarse_starts,
        speeds_mps=coarse_grid.speeds_mps,
        azimuths_deg=coarse_grid.azimuths_deg,
        top_k=coarse_top_k,
        use_weighted_scoring=use_weighted_scoring,
    )
    route_duration_s = float(profile.timestamps_s[-1] - profile.timestamps_s[0])
    seed_radius_m = max(refine_radius_m, max_speed_mps * route_duration_s * 0.5 + refine_radius_m)
    if search_radius_m is not None and search_radius_m > 0:
        seed_radius_m = min(seed_radius_m, search_radius_m)
    seed_step_m = max(refine_start_step_m * 2.0, min(coarse_start_step_m * 0.25, 2_500.0))
    seed_sweep_starts = make_refine_start_points(
        center_x_m,
        center_y_m,
        search_bounds,
        seed_radius_m,
        seed_step_m,
    )
    seed_sweep_candidates = search_candidates_rust(
        dem=raster.heights,
        transform=raster.transform,
        measured_profile_m=coarse_profile.terrain_profile_m,
        timestamps_s=coarse_profile.timestamps_s,
        start_points_m=seed_sweep_starts,
        speeds_mps=coarse_grid.speeds_mps,
        azimuths_deg=coarse_grid.azimuths_deg,
        top_k=max(coarse_top_k, 100),
        use_weighted_scoring=use_weighted_scoring,
    )
    seed_points = [[center_x_m, center_y_m]]
    if previous_x_m is not None and previous_y_m is not None:
        seed_points.append([float(previous_x_m), float(previous_y_m)])
    seed_candidates = search_candidates_rust(
        dem=raster.heights,
        transform=raster.transform,
        measured_profile_m=coarse_profile.terrain_profile_m,
        timestamps_s=coarse_profile.timestamps_s,
        start_points_m=np.asarray(seed_points, dtype=np.float64),
        speeds_mps=coarse_grid.speeds_mps,
        azimuths_deg=coarse_grid.azimuths_deg,
        top_k=len(seed_points),
        use_weighted_scoring=use_weighted_scoring,
    )
    for candidate in seed_candidates:
        candidate["force_wide_azimuth"] = 1
        candidate["force_wide_speed"] = 1
    coarse_candidates = _unique_candidates_by_start(coarse_candidates + seed_sweep_candidates + seed_candidates)
    if not coarse_candidates:
        raise ValueError("No valid coarse trajectory candidates found inside DEM bounds")

    best: dict[str, float | int] | None = None
    best_speeds = np.array([], dtype=np.float64)
    best_azimuths = np.array([], dtype=np.float64)

    for candidate in coarse_candidates:
        if int(candidate.get("force_wide_speed", 0)):
            speeds = np.arange(min_speed_mps, max_speed_mps + fine_speed_step_mps * 0.5, fine_speed_step_mps)
        else:
            speeds = make_local_values(
                float(candidate["speed"]),
                coarse_speed_step_mps,
                fine_speed_step_mps,
                min_speed_mps,
                max_speed_mps,
            )
        if int(candidate.get("force_wide_azimuth", 0)):
            azimuths = np.arange(0.0, 360.0, fine_azimuth_step_deg, dtype=np.float64)
        else:
            azimuths = make_local_azimuths(
                float(candidate["azimuth"]),
                coarse_azimuth_step_deg,
                fine_azimuth_step_deg,
            )
        starts = make_refine_start_points(
            float(candidate["start_x"]),
            float(candidate["start_y"]),
            search_bounds,
            refine_radius_m,
            refine_start_step_m,
        )
        refined_candidates = search_candidates_rust(
            dem=raster.heights,
            transform=raster.transform,
            measured_profile_m=profile.terrain_profile_m,
            timestamps_s=profile.timestamps_s,
            start_points_m=starts,
            speeds_mps=speeds,
            azimuths_deg=azimuths,
            top_k=1,
            use_weighted_scoring=use_weighted_scoring,
        )
        if not refined_candidates:
            continue

        refined = refined_candidates[0]
        if best is None or _candidate_score(refined) < _candidate_score(best):
            best = refined
            best_speeds = speeds
            best_azimuths = azimuths

    if best is None:
        raise ValueError("No valid refined trajectory candidates found")

    micro_radius_m = max(refine_start_step_m, 500.0)
    micro_step_m = max(50.0, min(100.0, refine_start_step_m / 5.0))
    micro_starts = make_refine_start_points(
        float(best["start_x"]),
        float(best["start_y"]),
        search_bounds,
        micro_radius_m,
        micro_step_m,
    )
    micro_speeds = make_local_values(
        float(best["speed"]),
        max(2.0, fine_speed_step_mps * 2.0),
        fine_speed_step_mps,
        min_speed_mps,
        max_speed_mps,
    )
    micro_azimuths = make_local_azimuths(
        float(best["azimuth"]),
        max(3.0, fine_azimuth_step_deg * 3.0),
        fine_azimuth_step_deg,
    )
    micro_candidates = search_candidates_rust(
        dem=raster.heights,
        transform=raster.transform,
        measured_profile_m=profile.terrain_profile_m,
        timestamps_s=profile.timestamps_s,
        start_points_m=micro_starts,
        speeds_mps=micro_speeds,
        azimuths_deg=micro_azimuths,
        top_k=1,
        use_weighted_scoring=use_weighted_scoring,
    )
    if micro_candidates and _candidate_score(micro_candidates[0]) <= _candidate_score(best):
        best = micro_candidates[0]
        best_speeds = micro_speeds
        best_azimuths = micro_azimuths

    nano_radius_m = max(100.0, micro_step_m * 2.0)
    nano_step_m = max(25.0, min(50.0, micro_step_m / 2.0))
    nano_starts = make_refine_start_points(
        float(best["start_x"]),
        float(best["start_y"]),
        search_bounds,
        nano_radius_m,
        nano_step_m,
    )
    nano_speed_step_mps = max(0.25, fine_speed_step_mps / 2.0)
    nano_azimuth_step_deg = max(0.25, fine_azimuth_step_deg / 2.0)
    nano_speeds = make_local_values(
        float(best["speed"]),
        max(1.0, fine_speed_step_mps),
        nano_speed_step_mps,
        min_speed_mps,
        max_speed_mps,
    )
    nano_azimuths = make_local_azimuths(
        float(best["azimuth"]),
        max(1.0, fine_azimuth_step_deg),
        nano_azimuth_step_deg,
    )
    nano_candidates = search_candidates_rust(
        dem=raster.heights,
        transform=raster.transform,
        measured_profile_m=profile.terrain_profile_m,
        timestamps_s=profile.timestamps_s,
        start_points_m=nano_starts,
        speeds_mps=nano_speeds,
        azimuths_deg=nano_azimuths,
        top_k=1,
        use_weighted_scoring=use_weighted_scoring,
    )
    if nano_candidates and _candidate_score(nano_candidates[0]) <= _candidate_score(best):
        best = nano_candidates[0]
        best_speeds = nano_speeds
        best_azimuths = nano_azimuths

    errors, correlations, best_flat_index = compute_error_grid_rust(
        dem=raster.heights,
        transform=raster.transform,
        measured_profile_m=profile.terrain_profile_m,
        timestamps_s=profile.timestamps_s,
        start_x_m=float(best["start_x"]),
        start_y_m=float(best["start_y"]),
        speeds_mps=best_speeds,
        azimuths_deg=best_azimuths,
        use_weighted_scoring=use_weighted_scoring,
    )
    best_speed_index, best_azimuth_index = np.unravel_index(best_flat_index, errors.shape)

    return _build_result_from_utm_raster(
        dem=raster.heights,
        transform=raster.transform,
        measured_profile_m=profile.terrain_profile_m,
        timestamps_s=profile.timestamps_s,
        speeds_mps=best_speeds,
        azimuths_deg=best_azimuths,
        start_x_m=float(best["start_x"]),
        start_y_m=float(best["start_y"]),
        speed_mps=float(best_speeds[best_speed_index]),
        azimuth_deg=float(best_azimuths[best_azimuth_index]),
        best_error=float(errors[best_speed_index, best_azimuth_index]),
        best_correlation=float(correlations[best_speed_index, best_azimuth_index]),
        best_speed_index=int(best_speed_index),
        best_azimuth_index=int(best_azimuth_index),
        errors=errors,
        correlations=correlations,
        terrain_variance_m2=terrain_variance,
        smoothing_window=smoothing_window,
        smoothing_method=smoothing_method,
        scoring_mode="weighted" if use_weighted_scoring else "regular",
    )


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


def estimate_accuracy_m(
    best_error_m: float,
    best_correlation: float,
    confidence: float,
    terrain_variance_m2: float,
    is_flat_terrain: bool = False,
) -> float:
    if is_flat_terrain:
        return float("inf")
    if not np.isfinite(best_error_m):
        return float("inf")

    correlation = best_correlation if np.isfinite(best_correlation) else 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    relief_factor = 1.0 / max(np.sqrt(max(terrain_variance_m2, 0.0)), 1.0)

    error_part = max(best_error_m * 2.0, 5.0)
    correlation_part = max(0.0, 1.0 - correlation) * 200.0
    confidence_part = max(0.0, 1.0 - confidence) * 100.0
    relief_part = relief_factor * 50.0
    return float(error_part + correlation_part + confidence_part + relief_part)


def classify_quality(estimated_accuracy_m: float, confidence: float, is_flat_terrain: bool) -> str:
    if is_flat_terrain or not np.isfinite(estimated_accuracy_m):
        return "низкая"
    if estimated_accuracy_m <= 50.0 and confidence >= 0.75:
        return "высокая"
    if estimated_accuracy_m <= 150.0 and confidence >= 0.45:
        return "средняя"
    return "низкая"


def _needs_unweighted_retry(result: LocalizationResult) -> bool:
    if result.is_flat_terrain:
        return False
    if result.confidence < 0.35:
        return True
    if np.isfinite(result.estimated_accuracy_m) and result.estimated_accuracy_m > 150.0:
        return True
    return False


def _needs_dense_retry(result: LocalizationResult, coarse_start_step_m: float, coarse_top_k: int) -> bool:
    if result.is_flat_terrain:
        return False
    if coarse_start_step_m <= 1_000.0 and coarse_top_k >= 200:
        return False
    if result.confidence < 0.20:
        return True
    if np.isfinite(result.estimated_accuracy_m) and result.estimated_accuracy_m > 150.0:
        return True
    return False


def _is_better_result(candidate: LocalizationResult, baseline: LocalizationResult) -> bool:
    if candidate.is_flat_terrain:
        return False
    if baseline.is_flat_terrain:
        return True

    candidate_accuracy = candidate.estimated_accuracy_m if np.isfinite(candidate.estimated_accuracy_m) else float("inf")
    baseline_accuracy = baseline.estimated_accuracy_m if np.isfinite(baseline.estimated_accuracy_m) else float("inf")

    if candidate.confidence >= baseline.confidence + 0.2:
        return True
    if candidate_accuracy < baseline_accuracy and candidate.confidence >= baseline.confidence - 0.05:
        return True
    if candidate_accuracy <= baseline_accuracy * 0.75 and candidate.confidence >= baseline.confidence:
        return True
    if candidate.best_correlation >= baseline.best_correlation + 0.02 and candidate_accuracy < baseline_accuracy:
        return True
    return False


def _replace_scoring_mode(result: LocalizationResult, scoring_mode: str) -> LocalizationResult:
    return replace(result, scoring_mode=scoring_mode)


def _clip_bounds_around_center(
    bounds: tuple[float, float, float, float],
    center: tuple[float, float],
    radius_m: float | None,
) -> tuple[float, float, float, float]:
    if radius_m is None or radius_m <= 0:
        return bounds

    min_x, max_x, min_y, max_y = bounds
    center_x, center_y = center
    return (
        max(min_x, center_x - radius_m),
        min(max_x, center_x + radius_m),
        max(min_y, center_y - radius_m),
        min(max_y, center_y + radius_m),
    )


def _unique_candidates_by_start(candidates: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    unique: dict[tuple[int, int], dict[str, float | int]] = {}
    for candidate in candidates:
        key = (round(float(candidate["start_x"])), round(float(candidate["start_y"])))
        old = unique.get(key)
        if old is None or _candidate_score(candidate) < _candidate_score(old):
            unique[key] = candidate
        elif int(candidate.get("force_wide_azimuth", 0)):
            old["force_wide_azimuth"] = 1
            old["force_wide_speed"] = 1
    return sorted(unique.values(), key=_candidate_score)


def _candidate_score(candidate: dict[str, float | int]) -> float:
    error = float(candidate.get("error", float("inf")))
    if not np.isfinite(error):
        return float("inf")

    correlation = float(candidate.get("correlation", -1.0))
    if not np.isfinite(correlation):
        correlation = -1.0
    correlation = float(np.clip(correlation, -1.0, 1.0))
    return error * (1.0 + 0.35 * max(0.0, 1.0 - correlation))


def _limit_profile_points(profile: NmeaProfile, max_points: int | None) -> NmeaProfile:
    if max_points is None or max_points <= 0 or profile.timestamps_s.size <= max_points:
        return profile

    indexes = np.unique(np.linspace(0, profile.timestamps_s.size - 1, int(max_points), dtype=int))
    return NmeaProfile(
        radio_altitudes_m=profile.radio_altitudes_m[indexes],
        terrain_profile_m=profile.terrain_profile_m[indexes],
        timestamps_s=profile.timestamps_s[indexes],
    )


def _select_informative_profile_points(profile: NmeaProfile, max_points: int | None) -> NmeaProfile:
    size = profile.timestamps_s.size
    if max_points is None or max_points <= 0 or size <= max_points:
        return profile

    max_points = int(max_points)
    heights = profile.terrain_profile_m.astype(np.float64, copy=False)
    fill = float(np.nanmedian(heights)) if np.isfinite(heights).any() else 0.0
    clean_heights = np.nan_to_num(heights, nan=fill)
    gradient = np.abs(np.gradient(clean_heights))
    curvature = np.abs(np.gradient(np.gradient(clean_heights)))
    scores = gradient + 0.5 * curvature

    mandatory: set[int] = {0, size - 1}
    base_count = min(size, max(2, max_points // 3))
    mandatory.update(int(index) for index in np.linspace(0, size - 1, base_count, dtype=int))

    selected = list(mandatory)
    for index in np.argsort(scores)[::-1]:
        selected.append(int(index))
        if len(set(selected)) >= max_points:
            break

    indexes = np.array(sorted(set(selected))[:max_points], dtype=int)
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
        velocity_x_mps=0.0,
        velocity_y_mps=0.0,
        start_x_m=x_m,
        start_y_m=y_m,
        current_x_m=x_m,
        current_y_m=y_m,
        best_error=float("nan"),
        best_correlation=float("nan"),
        best_speed_index=-1,
        best_azimuth_index=-1,
        errors=errors,
        correlations=np.full_like(errors, np.nan, dtype=np.float64),
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
        estimated_accuracy_m=float("inf"),
        quality_label="низкая",
        scoring_mode="flat",
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
    best_correlation: float,
    best_speed_index: int,
    best_azimuth_index: int,
    errors: np.ndarray,
    correlations: np.ndarray,
    terrain_variance_m2: float,
    smoothing_window: int,
    smoothing_method: str,
    scoring_mode: str,
) -> LocalizationResult:
    trajectory_x, trajectory_y = compute_trajectory_points(start_x_m, start_y_m, speed_mps, azimuth_deg, timestamps_s)
    predicted = sample_dem_heights(dem, dataset, source_crs, utm_crs, trajectory_x, trajectory_y)
    smoothed_x, smoothed_y = _smooth_trajectory(trajectory_x, trajectory_y, timestamps_s, smoothing_window, smoothing_method)
    velocity_x_mps, velocity_y_mps = compute_velocity_components(speed_mps, azimuth_deg)
    confidence = compute_confidence(errors)
    estimated_accuracy_m = estimate_accuracy_m(best_error, best_correlation, confidence, terrain_variance_m2)

    return LocalizationResult(
        speed_mps=speed_mps,
        azimuth_deg=azimuth_deg,
        velocity_x_mps=velocity_x_mps,
        velocity_y_mps=velocity_y_mps,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        current_x_m=float(trajectory_x[-1]),
        current_y_m=float(trajectory_y[-1]),
        best_error=best_error,
        best_correlation=best_correlation,
        best_speed_index=best_speed_index,
        best_azimuth_index=best_azimuth_index,
        errors=errors,
        correlations=correlations,
        speeds_mps=speeds_mps,
        azimuths_deg=azimuths_deg,
        trajectory_x_m=trajectory_x,
        trajectory_y_m=trajectory_y,
        measured_profile_m=measured_profile_m,
        predicted_profile_m=predicted,
        smoothed_trajectory_x_m=smoothed_x,
        smoothed_trajectory_y_m=smoothed_y,
        terrain_variance_m2=terrain_variance_m2,
        confidence=confidence,
        estimated_accuracy_m=estimated_accuracy_m,
        quality_label=classify_quality(estimated_accuracy_m, confidence, False),
        scoring_mode=scoring_mode,
        is_flat_terrain=False,
    )


def _build_result_from_utm_raster(
    dem: np.ndarray,
    transform,
    measured_profile_m: np.ndarray,
    timestamps_s: np.ndarray,
    speeds_mps: np.ndarray,
    azimuths_deg: np.ndarray,
    start_x_m: float,
    start_y_m: float,
    speed_mps: float,
    azimuth_deg: float,
    best_error: float,
    best_correlation: float,
    best_speed_index: int,
    best_azimuth_index: int,
    errors: np.ndarray,
    correlations: np.ndarray,
    terrain_variance_m2: float,
    smoothing_window: int,
    smoothing_method: str,
    scoring_mode: str,
) -> LocalizationResult:
    trajectory_x, trajectory_y = compute_trajectory_points(start_x_m, start_y_m, speed_mps, azimuth_deg, timestamps_s)
    predicted = sample_utm_raster_heights(dem, transform, trajectory_x, trajectory_y)
    smoothed_x, smoothed_y = _smooth_trajectory(trajectory_x, trajectory_y, timestamps_s, smoothing_window, smoothing_method)
    velocity_x_mps, velocity_y_mps = compute_velocity_components(speed_mps, azimuth_deg)
    confidence = compute_confidence(errors)
    estimated_accuracy_m = estimate_accuracy_m(best_error, best_correlation, confidence, terrain_variance_m2)

    return LocalizationResult(
        speed_mps=speed_mps,
        azimuth_deg=azimuth_deg,
        velocity_x_mps=velocity_x_mps,
        velocity_y_mps=velocity_y_mps,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        current_x_m=float(trajectory_x[-1]),
        current_y_m=float(trajectory_y[-1]),
        best_error=best_error,
        best_correlation=best_correlation,
        best_speed_index=best_speed_index,
        best_azimuth_index=best_azimuth_index,
        errors=errors,
        correlations=correlations,
        speeds_mps=speeds_mps,
        azimuths_deg=azimuths_deg,
        trajectory_x_m=trajectory_x,
        trajectory_y_m=trajectory_y,
        measured_profile_m=measured_profile_m,
        predicted_profile_m=predicted,
        smoothed_trajectory_x_m=smoothed_x,
        smoothed_trajectory_y_m=smoothed_y,
        terrain_variance_m2=terrain_variance_m2,
        confidence=confidence,
        estimated_accuracy_m=estimated_accuracy_m,
        quality_label=classify_quality(estimated_accuracy_m, confidence, False),
        scoring_mode=scoring_mode,
        is_flat_terrain=False,
    )


def compute_velocity_components(speed_mps: float, azimuth_deg: float) -> tuple[float, float]:
    azimuth_rad = np.deg2rad(azimuth_deg)
    return float(speed_mps * np.sin(azimuth_rad)), float(speed_mps * np.cos(azimuth_rad))


def _smooth_trajectory(
    trajectory_x: np.ndarray,
    trajectory_y: np.ndarray,
    timestamps_s: np.ndarray,
    smoothing_window: int,
    smoothing_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if smoothing_method == "moving_average":
        return smooth_trajectory(trajectory_x, trajectory_y, window=smoothing_window)
    return kalman_smooth_trajectory(trajectory_x, trajectory_y, timestamps_s)
