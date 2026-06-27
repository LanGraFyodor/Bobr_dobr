from __future__ import annotations

import numpy as np
import rasterio
from pyproj import CRS

from terrain_nav.core.sampling import sample_dem_heights


def compute_error_grid(
    dem: np.ndarray,
    dataset: rasterio.io.DatasetReader,
    source_crs: CRS,
    utm_crs: CRS,
    measured_profile_m: np.ndarray,
    timestamps_s: np.ndarray,
    start_x_m: float,
    start_y_m: float,
    speeds_mps: np.ndarray,
    azimuths_deg: np.ndarray,
    use_weighted_scoring: bool = True,
) -> tuple[np.ndarray, int]:
    time_offsets = timestamps_s - timestamps_s[0]

    speeds = speeds_mps[:, None, None]
    azimuths_rad = np.deg2rad(azimuths_deg)[None, :, None]
    distances = speeds * time_offsets[None, None, :]

    x = start_x_m + distances * np.sin(azimuths_rad)
    y = start_y_m + distances * np.cos(azimuths_rad)

    predicted = sample_dem_heights(
        dem=dem,
        dataset=dataset,
        source_crs=source_crs,
        utm_crs=utm_crs,
        x_m=x.ravel(),
        y_m=y.ravel(),
    ).reshape(speeds_mps.size, azimuths_deg.size, measured_profile_m.size)

    weights = make_profile_weights(measured_profile_m) if use_weighted_scoring else np.ones_like(measured_profile_m)
    diff = predicted - measured_profile_m[None, None, :]
    valid = np.isfinite(diff)
    valid_counts = np.sum(valid, axis=2)
    squared_error_sum = np.nansum(diff * diff * weights[None, None, :], axis=2)
    errors = np.full(valid_counts.shape, np.inf, dtype=np.float64)
    complete = valid_counts == measured_profile_m.size
    errors[complete] = np.sqrt(squared_error_sum[complete] / np.sum(weights))

    return errors, int(np.argmin(errors))


def make_profile_weights(profile_m: np.ndarray) -> np.ndarray:
    if profile_m.size < 3:
        return np.ones(profile_m.shape, dtype=np.float64)

    fill = float(np.nanmedian(profile_m)) if np.isfinite(profile_m).any() else 0.0
    clean = np.nan_to_num(profile_m.astype(np.float64, copy=False), nan=fill)
    gradient = np.abs(np.gradient(clean))
    curvature = np.abs(np.gradient(np.gradient(clean)))
    features = gradient + 0.5 * curvature
    feature_mean = float(np.mean(features))
    if not np.isfinite(feature_mean) or feature_mean <= 1e-9:
        return np.ones(profile_m.shape, dtype=np.float64)

    return 1.0 + np.clip(features / feature_mean, 0.0, 4.0)


def search_start_points(
    dem: np.ndarray,
    dataset: rasterio.io.DatasetReader,
    source_crs: CRS,
    utm_crs: CRS,
    measured_profile_m: np.ndarray,
    timestamps_s: np.ndarray,
    start_points: np.ndarray,
    speeds_mps: np.ndarray,
    azimuths_deg: np.ndarray,
    top_k: int = 1,
    use_weighted_scoring: bool = True,
) -> dict[str, float | int | np.ndarray | list[dict[str, float | int | np.ndarray]]]:
    best: dict[str, float | int | np.ndarray] | None = None
    candidates: list[dict[str, float | int | np.ndarray]] = []

    for start_x_m, start_y_m in start_points:
        errors, flat_index = compute_error_grid(
            dem,
            dataset,
            source_crs,
            utm_crs,
            measured_profile_m,
            timestamps_s,
            float(start_x_m),
            float(start_y_m),
            speeds_mps,
            azimuths_deg,
            use_weighted_scoring=use_weighted_scoring,
        )
        candidate_error = float(errors.ravel()[flat_index])
        if not np.isfinite(candidate_error):
            continue

        speed_index, azimuth_index = np.unravel_index(flat_index, errors.shape)
        candidate = {
            "start_x": float(start_x_m),
            "start_y": float(start_y_m),
            "speed": float(speeds_mps[speed_index]),
            "azimuth": float(azimuths_deg[azimuth_index]),
            "error": candidate_error,
            "speed_index": int(speed_index),
            "azimuth_index": int(azimuth_index),
            "errors": errors,
        }
        candidates.append(candidate)
        if best is None or candidate_error < float(best["error"]):
            best = candidate

    if best is None:
        raise ValueError("No valid trajectory candidates found inside DEM bounds")

    return {**best, "candidates": sorted(candidates, key=lambda item: float(item["error"]))[: max(1, top_k)]}


def refine_best_candidates(
    coarse_candidates: list[dict[str, float | int | np.ndarray]],
    bounds: tuple[float, float, float, float],
    dem: np.ndarray,
    dataset: rasterio.io.DatasetReader,
    source_crs: CRS,
    utm_crs: CRS,
    measured_profile_m: np.ndarray,
    timestamps_s: np.ndarray,
    min_speed_mps: float,
    max_speed_mps: float,
    coarse_speed_step_mps: float,
    fine_speed_step_mps: float,
    coarse_azimuth_step_deg: float,
    fine_azimuth_step_deg: float,
    refine_radius_m: float,
    refine_start_step_m: float,
    use_weighted_scoring: bool = True,
) -> tuple[dict[str, float | int | np.ndarray], np.ndarray, np.ndarray]:
    best: dict[str, float | int | np.ndarray] | None = None
    best_speeds = np.array([], dtype=np.float64)
    best_azimuths = np.array([], dtype=np.float64)

    for candidate in coarse_candidates:
        speeds = make_local_values(
            float(candidate["speed"]),
            coarse_speed_step_mps,
            fine_speed_step_mps,
            min_speed_mps,
            max_speed_mps,
        )
        azimuths = make_local_azimuths(
            float(candidate["azimuth"]),
            coarse_azimuth_step_deg,
            fine_azimuth_step_deg,
        )
        starts = make_refine_start_points(
            float(candidate["start_x"]),
            float(candidate["start_y"]),
            bounds,
            refine_radius_m,
            refine_start_step_m,
        )

        refined = search_start_points(
            dem=dem,
            dataset=dataset,
            source_crs=source_crs,
            utm_crs=utm_crs,
            measured_profile_m=measured_profile_m,
            timestamps_s=timestamps_s,
            start_points=starts,
            speeds_mps=speeds,
            azimuths_deg=azimuths,
            use_weighted_scoring=use_weighted_scoring,
        )
        if best is None or float(refined["error"]) < float(best["error"]):
            best = refined
            best_speeds = speeds
            best_azimuths = azimuths

    if best is None:
        raise ValueError("No valid refined trajectory candidates found")

    return best, best_speeds, best_azimuths


def make_start_points(
    bounds: tuple[float, float, float, float],
    step_m: float,
    extra_points: np.ndarray | None = None,
) -> np.ndarray:
    if step_m <= 0:
        raise ValueError("step_m must be positive")

    min_x, max_x, min_y, max_y = bounds
    xs = np.arange(min_x, max_x + step_m * 0.5, step_m, dtype=np.float64)
    ys = np.arange(min_y, max_y + step_m * 0.5, step_m, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    if extra_points is not None and extra_points.size:
        points = np.vstack([points, extra_points])

    return np.unique(np.round(points, decimals=3), axis=0)


def make_refine_start_points(
    center_x_m: float,
    center_y_m: float,
    bounds: tuple[float, float, float, float],
    radius_m: float,
    step_m: float,
) -> np.ndarray:
    if radius_m < 0:
        raise ValueError("radius_m must be non-negative")

    min_x, max_x, min_y, max_y = bounds
    xs = np.arange(center_x_m - radius_m, center_x_m + radius_m + step_m * 0.5, step_m)
    ys = np.arange(center_y_m - radius_m, center_y_m + radius_m + step_m * 0.5, step_m)
    xs = xs[(xs >= min_x) & (xs <= max_x)]
    ys = ys[(ys >= min_y) & (ys <= max_y)]

    if xs.size == 0:
        xs = np.array([np.clip(center_x_m, min_x, max_x)], dtype=np.float64)
    if ys.size == 0:
        ys = np.array([np.clip(center_y_m, min_y, max_y)], dtype=np.float64)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    center = np.array([[np.clip(center_x_m, min_x, max_x), np.clip(center_y_m, min_y, max_y)]])
    return np.unique(np.round(np.vstack([points, center]), decimals=3), axis=0)


def make_local_values(center: float, radius: float, step: float, min_value: float, max_value: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")

    start = max(min_value, center - radius)
    stop = min(max_value, center + radius)
    values = np.arange(start, stop + step * 0.5, step, dtype=np.float64)
    values = np.append(values, np.clip(center, min_value, max_value))
    return np.unique(np.round(values, decimals=6))


def make_local_azimuths(center: float, radius_deg: float, step_deg: float) -> np.ndarray:
    if step_deg <= 0:
        raise ValueError("step_deg must be positive")

    offsets = np.arange(-radius_deg, radius_deg + step_deg * 0.5, step_deg, dtype=np.float64)
    azimuths = (center + offsets) % 360.0
    azimuths = np.append(azimuths, center % 360.0)
    return np.unique(np.round(azimuths, decimals=6))
