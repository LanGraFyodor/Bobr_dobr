from __future__ import annotations

import numpy as np
import rasterio
from pyproj import CRS
from pathlib import Path

from terrain_nav.dem import dataset_center_utm, dataset_utm_bounds
from terrain_nav.search import _load_dem_context, make_search_grid, _limit_profile_points, compute_confidence
from terrain_nav.matching import make_start_points, search_start_points, refine_best_candidates
from terrain_nav.models import LocalizationResult, NmeaProfile
from terrain_nav.nmea import parse_nmea_profile
from terrain_nav.kalman import KalmanFilter2D

def localize_sliding_window(
    dem_path: str | Path,
    nmea_path: str | Path,
    baro_altitude_m: float = 1500.0,
    sample_rate_hz: float = 1.0,
    window_size_s: float = 15.0,
    step_size_s: float = 5.0,
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
        bounds = dataset_utm_bounds(dataset, source_crs, utm_crs)

        coarse_grid = make_search_grid(
            min_speed_mps=min_speed_mps,
            max_speed_mps=max_speed_mps,
            speed_step_mps=coarse_speed_step_mps,
            azimuth_step_deg=coarse_azimuth_step_deg,
        )

        n_samples = profile.timestamps_s.size
        window_points = max(3, int(window_size_s * sample_rate_hz))
        step_points = max(1, int(step_size_s * sample_rate_hz))

        kf = None
        trajectory_x = []
        trajectory_y = []
        
        last_errors = None
        last_fine_speeds = None
        last_fine_azimuths = None
        last_best_error = float('inf')
        last_best_speed_idx = 0
        last_best_azimuth_idx = 0

        # Run sliding window
        for start_idx in range(0, n_samples, step_points):
            end_idx = start_idx + window_points
            if end_idx > n_samples:
                if start_idx == 0:
                    end_idx = n_samples
                else:
                    break

            win_timestamps = profile.timestamps_s[start_idx:end_idx]
            win_profile = profile.terrain_profile_m[start_idx:end_idx]
            
            terrain_variance = float(np.nanvar(win_profile))
            
            dt = 0.0
            if start_idx > 0:
                dt = profile.timestamps_s[start_idx] - profile.timestamps_s[start_idx - step_points]
            
            if kf is not None:
                kf.predict(dt, process_noise_std=2.0)
                current_x = kf.x
                current_y = kf.y
                # Save predicted trajectory for points between last update and now
                for p in range(step_points):
                    if start_idx - step_points + p < n_samples:
                        t_diff = profile.timestamps_s[start_idx - step_points + p] - profile.timestamps_s[start_idx - step_points]
                        trajectory_x.append(current_x - kf.vx * (dt - t_diff))
                        trajectory_y.append(current_y - kf.vy * (dt - t_diff))
            else:
                for p in range(step_points):
                    if start_idx - step_points + p >= 0:
                        trajectory_x.append(center_x_m)
                        trajectory_y.append(center_y_m)

            if terrain_variance <= flat_variance_threshold_m2:
                continue

            if kf is None:
                # Global Search
                coarse_starts = make_start_points(
                    bounds=bounds,
                    step_m=coarse_start_step_m,
                    extra_points=np.array([[center_x_m, center_y_m]], dtype=np.float64),
                )
            else:
                # Local Search around prediction
                coarse_starts = make_start_points(
                    bounds=bounds,
                    step_m=refine_start_step_m * 2,
                    extra_points=np.array([[kf.x, kf.y]], dtype=np.float64),
                )
                
            coarse = search_start_points(
                dem=dem,
                dataset=dataset,
                source_crs=source_crs,
                utm_crs=utm_crs,
                measured_profile_m=win_profile,
                timestamps_s=win_timestamps,
                start_points=coarse_starts,
                speeds_mps=coarse_grid.speeds_mps,
                azimuths_deg=coarse_grid.azimuths_deg,
                top_k=coarse_top_k if kf is None else max(1, coarse_top_k // 2),
            )

            fine, fine_speeds, fine_azimuths = refine_best_candidates(
                coarse_candidates=coarse["candidates"],
                bounds=bounds,
                dem=dem,
                dataset=dataset,
                source_crs=source_crs,
                utm_crs=utm_crs,
                measured_profile_m=win_profile,
                timestamps_s=win_timestamps,
                min_speed_mps=min_speed_mps,
                max_speed_mps=max_speed_mps,
                coarse_speed_step_mps=coarse_speed_step_mps,
                fine_speed_step_mps=fine_speed_step_mps,
                coarse_azimuth_step_deg=coarse_azimuth_step_deg,
                fine_azimuth_step_deg=fine_azimuth_step_deg,
                refine_radius_m=refine_radius_m,
                refine_start_step_m=refine_start_step_m,
            )

            best_x = float(fine["start_x"])
            best_y = float(fine["start_y"])
            best_speed = float(fine["speed"])
            best_azimuth = float(fine["azimuth"])
            
            last_errors = fine["errors"]
            last_fine_speeds = fine_speeds
            last_fine_azimuths = fine_azimuths
            last_best_error = float(fine["error"])
            last_best_speed_idx = int(fine["speed_index"])
            last_best_azimuth_idx = int(fine["azimuth_index"])

            if kf is None:
                az_rad = np.deg2rad(best_azimuth)
                vx = best_speed * np.sin(az_rad)
                vy = best_speed * np.cos(az_rad)
                kf = KalmanFilter2D(best_x, best_y, vx, vy)
                # Overwrite empty start trajectory
                trajectory_x = [best_x] * step_points
                trajectory_y = [best_y] * step_points
            else:
                kf.update(best_x, best_y, measurement_noise_std=10.0)
                
        # Fill remaining
        while len(trajectory_x) < n_samples:
            trajectory_x.append(kf.x if kf else center_x_m)
            trajectory_y.append(kf.y if kf else center_y_m)

        trajectory_x = np.array(trajectory_x)
        trajectory_y = np.array(trajectory_y)

        # fallback grid
        if last_errors is None:
            last_errors = np.full((1, 1), np.nan)
            last_fine_speeds = np.array([0.0])
            last_fine_azimuths = np.array([0.0])
            
        from terrain_nav.sampling import sample_dem_heights
        predicted_profile = sample_dem_heights(dem, dataset, source_crs, utm_crs, trajectory_x, trajectory_y)
        
        from terrain_nav.geometry import smooth_trajectory
        smoothed_x, smoothed_y = smooth_trajectory(trajectory_x, trajectory_y, window=smoothing_window)

        return LocalizationResult(
            speed_mps=kf.speed if kf else 0.0,
            azimuth_deg=kf.azimuth_deg if kf else 0.0,
            start_x_m=trajectory_x[0],
            start_y_m=trajectory_y[0],
            current_x_m=trajectory_x[-1],
            current_y_m=trajectory_y[-1],
            best_error=last_best_error,
            best_speed_index=last_best_speed_idx,
            best_azimuth_index=last_best_azimuth_idx,
            errors=last_errors,
            speeds_mps=last_fine_speeds,
            azimuths_deg=last_fine_azimuths,
            trajectory_x_m=trajectory_x,
            trajectory_y_m=trajectory_y,
            measured_profile_m=profile.terrain_profile_m,
            predicted_profile_m=predicted_profile,
            smoothed_trajectory_x_m=smoothed_x,
            smoothed_trajectory_y_m=smoothed_y,
            terrain_variance_m2=float(np.nanvar(profile.terrain_profile_m)),
            confidence=compute_confidence(last_errors),
            is_flat_terrain=(kf is None),
        )
