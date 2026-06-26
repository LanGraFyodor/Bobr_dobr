from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS

from terrain_nav.dem import dataset_center_lonlat, dataset_center_utm, dataset_utm_bounds, utm_crs_for_lonlat
from terrain_nav.geometry import compute_trajectory_points
from terrain_nav.models import GeneratedFlight
from terrain_nav.sampling import sample_dem_heights


def generate_test_flight(
    dem_path: str | Path,
    output_path: str | Path,
    start_x_m: float | None = None,
    start_y_m: float | None = None,
    speed_mps: float = 20.0,
    azimuth_deg: float = 45.0,
    duration_s: float = 2400.0,
    sample_rate_hz: float = 1.0,
    baro_altitude_m: float = 1500.0,
    noise_std_m: float = 2.0,
    seed: int = 42,
) -> GeneratedFlight:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as dataset:
        dem = dataset.read(1).astype(np.float64)
        if dataset.nodata is not None:
            dem[dem == dataset.nodata] = np.nan

        source_crs = CRS.from_user_input(dataset.crs)
        center_lon, center_lat = dataset_center_lonlat(dataset, source_crs)
        utm_crs = utm_crs_for_lonlat(center_lon, center_lat)

        timestamps_s = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
        if start_x_m is None or start_y_m is None:
            start_x_m, start_y_m = _auto_start_for_trajectory(
                dataset=dataset,
                source_crs=source_crs,
                utm_crs=utm_crs,
                speed_mps=speed_mps,
                azimuth_deg=azimuth_deg,
                timestamps_s=timestamps_s,
            )

        trajectory_x_m, trajectory_y_m = compute_trajectory_points(
            start_x_m=float(start_x_m),
            start_y_m=float(start_y_m),
            speed_mps=speed_mps,
            azimuth_deg=azimuth_deg,
            timestamps_s=timestamps_s,
        )
        terrain_heights_m = sample_dem_heights(dem, dataset, source_crs, utm_crs, trajectory_x_m, trajectory_y_m)

    if np.isnan(terrain_heights_m).any():
        raise ValueError("Generated trajectory leaves DEM bounds or crosses nodata cells")

    rng = np.random.default_rng(seed)
    radio_altitudes_m = baro_altitude_m - terrain_heights_m
    noisy_radio_altitudes_m = radio_altitudes_m + rng.normal(0.0, noise_std_m, radio_altitudes_m.shape)
    nmea_lines = [make_gga_sentence(t, altitude) for t, altitude in zip(timestamps_s, noisy_radio_altitudes_m)]
    output_path.write_text("\n".join(nmea_lines) + "\n", encoding="utf-8")

    return GeneratedFlight(
        timestamps_s=timestamps_s,
        trajectory_x_m=trajectory_x_m,
        trajectory_y_m=trajectory_y_m,
        terrain_heights_m=terrain_heights_m,
        radio_altitudes_m=radio_altitudes_m,
        noisy_radio_altitudes_m=noisy_radio_altitudes_m,
        nmea_lines=nmea_lines,
    )


def make_gga_sentence(timestamp_s: float, altitude_m: float) -> str:
    payload = f"GPGGA,{format_nmea_time(timestamp_s)},,,,,,,,{altitude_m:.3f},M,0.0,M,,"
    checksum = 0
    for char in payload:
        checksum ^= ord(char)
    return f"${payload}*{checksum:02X}"


def format_nmea_time(timestamp_s: float) -> str:
    timestamp_s %= 24.0 * 3600.0
    hours = int(timestamp_s // 3600.0)
    minutes = int((timestamp_s % 3600.0) // 60.0)
    seconds = timestamp_s - hours * 3600.0 - minutes * 60.0
    return f"{hours:02d}{minutes:02d}{seconds:06.3f}"


def _auto_start_for_trajectory(
    dataset: rasterio.io.DatasetReader,
    source_crs: CRS,
    utm_crs: CRS,
    speed_mps: float,
    azimuth_deg: float,
    timestamps_s: np.ndarray,
) -> tuple[float, float]:
    if timestamps_s.size == 0:
        return dataset_center_utm(dataset, source_crs, utm_crs)

    bounds = dataset_utm_bounds(dataset, source_crs, utm_crs)
    min_x, max_x, min_y, max_y = bounds

    total_time_s = float(timestamps_s[-1] - timestamps_s[0])
    distance_m = float(speed_mps) * total_time_s
    azimuth_rad = np.deg2rad(azimuth_deg)
    end_dx_m = distance_m * np.sin(azimuth_rad)
    end_dy_m = distance_m * np.cos(azimuth_rad)

    margin_m = min(2_000.0, max((max_x - min_x), (max_y - min_y)) * 0.03)
    start_min_x = min_x + margin_m - min(0.0, end_dx_m)
    start_max_x = max_x - margin_m - max(0.0, end_dx_m)
    start_min_y = min_y + margin_m - min(0.0, end_dy_m)
    start_max_y = max_y - margin_m - max(0.0, end_dy_m)

    if start_min_x > start_max_x or start_min_y > start_max_y:
        max_route_m = min(max_x - min_x, max_y - min_y) - 2.0 * margin_m
        raise ValueError(
            "Generated trajectory is longer than the DEM allows. "
            f"Reduce speed/duration: route is {distance_m:.0f} m, safe map span is about {max_route_m:.0f} m."
        )

    return (float((start_min_x + start_max_x) * 0.5), float((start_min_y + start_max_y) * 0.5))


_format_nmea_time = format_nmea_time
