from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import rasterio

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from terrain_nav.search import localize_from_nmea, localize_position_from_nmea
from terrain_nav.dem import dataset_center_lonlat, utm_crs_for_lonlat


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize terrain navigation algorithm results")
    parser.add_argument("--dem", default=str(ROOT_DIR / "data" / "map.tif"))
    parser.add_argument("--nmea", default=str(ROOT_DIR / "outputs" / "test_flight.nmea"))
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "outputs"))
    parser.add_argument("--start-x", type=float, default=None, help="UTM X start coordinate. If None, uses center of map.")
    parser.add_argument("--start-y", type=float, default=None, help="UTM Y start coordinate. If None, uses center of map.")
    parser.add_argument("--full-search", action="store_true", help="Perform full X,Y search instead of center start.")
    parser.add_argument("--baro", type=float, default=1500.0, help="Constant barometric altitude (default: 1500.0)")
    return parser


def plot_heatmap(result, out_path: Path):
    # errors array is 1.0 - correlation, so correlation is 1.0 - errors
    correlations = 1.0 - result.errors
    
    # Check if the array is fully empty (e.g. flat terrain fallback)
    if np.all(np.isnan(correlations)):
        print("Cannot plot heatmap: flat terrain or no valid correlation data.")
        return

    speeds, azimuths = np.meshgrid(result.azimuths_deg, result.speeds_mps)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(speeds, azimuths, correlations, shading='auto', cmap='viridis')
    plt.colorbar(label='Pearson Correlation Coefficient')
    
    # Mark the best spot
    plt.scatter([result.azimuth_deg], [result.speed_mps], color='red', marker='x', s=100, label='Best Match')
    
    plt.title('Correlation Heatmap: Speed vs Azimuth')
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_trajectory_on_map(dem_path: str, result, out_path: Path):
    with rasterio.open(dem_path) as dataset:
        dem = dataset.read(1).astype(np.float64)
        if dataset.nodata is not None:
            dem[dem == dataset.nodata] = np.nan
            
        bounds = dataset.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
    plt.figure(figsize=(10, 10))
    plt.imshow(dem, extent=extent, origin='upper', cmap='terrain', alpha=0.8)
    plt.colorbar(label='Elevation (m)')
    
    # Plot predicted trajectory
    plt.plot(result.trajectory_x_m, result.trajectory_y_m, color='red', linewidth=2, label='Estimated Trajectory')
    plt.scatter(result.trajectory_x_m[0], result.trajectory_y_m[0], color='blue', marker='o', s=50, label='Start')
    plt.scatter(result.trajectory_x_m[-1], result.trajectory_y_m[-1], color='red', marker='x', s=50, label='End')
    
    plt.title('Estimated Flight Trajectory on DEM')
    plt.xlabel('UTM X (m)')
    plt.ylabel('UTM Y (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_profile_comparison(result, out_path: Path):
    plt.figure(figsize=(12, 5))
    
    # Smooth out the measured profile for visual comparison
    from scipy.signal import medfilt
    smoothed_meas = medfilt(result.measured_profile_m, kernel_size=5)
    
    plt.plot(smoothed_meas, label='Measured Profile (Altimeter)', alpha=0.7)
    plt.plot(result.predicted_profile_m, label='Predicted Profile (DEM)', alpha=0.7)
    
    plt.title('Terrain Profile Match')
    plt.xlabel('Sample Index (Time)')
    plt.ylabel('Absolute Elevation (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = build_arg_parser().parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running localization algorithm...")
    
    if args.full_search:
        result = localize_position_from_nmea(
            dem_path=args.dem,
            nmea_path=args.nmea,
            baro_altitude_m=args.baro,
        )
    else:
        result = localize_from_nmea(
            dem_path=args.dem,
            nmea_path=args.nmea,
            start_x_m=args.start_x,
            start_y_m=args.start_y,
            baro_altitude_m=args.baro,
        )
        
    print(f"Algorithm finished. Best speed: {result.speed_mps:.1f} m/s, Azimuth: {result.azimuth_deg:.1f} deg")
    
    heatmap_path = out_dir / "correlation_heatmap.png"
    traj_path = out_dir / "trajectory_map.png"
    prof_path = out_dir / "profile_comparison.png"
    
    print(f"Generating heatmap -> {heatmap_path}")
    plot_heatmap(result, heatmap_path)
    
    print(f"Generating trajectory map -> {traj_path}")
    plot_trajectory_on_map(args.dem, result, traj_path)
    
    print(f"Generating profile comparison -> {prof_path}")
    plot_profile_comparison(result, prof_path)
    
    print("Visualization complete.")

if __name__ == "__main__":
    main()
