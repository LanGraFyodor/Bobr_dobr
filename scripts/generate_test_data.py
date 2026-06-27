from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from terrain_nav.simulation import generate_test_flight


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fake radio-altimeter NMEA from DEM")
    parser.add_argument("--dem", default=str(ROOT_DIR / "data" / "map.tif"))
    parser.add_argument("--out", default=str(ROOT_DIR / "outputs" / "test_flight.nmea"))
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--azimuth", type=float, default=225.0)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--baro-altitude", type=float, default=1500.0)
    parser.add_argument("--noise-std", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--route-seed", type=int, default=7)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    generated = generate_test_flight(
        dem_path=args.dem,
        output_path=args.out,
        start_x_m=args.start_x,
        start_y_m=args.start_y,
        speed_mps=args.speed,
        azimuth_deg=args.azimuth,
        duration_s=args.duration,
        sample_rate_hz=args.rate,
        baro_altitude_m=args.baro_altitude,
        noise_std_m=args.noise_std,
        seed=args.seed,
        route_seed=args.route_seed,
    )

    print(f"saved: {args.out}")
    print(f"samples: {generated.timestamps_s.size}")
    print(f"speed_mps: {args.speed}")
    print(f"azimuth_deg: {args.azimuth}")
    print(f"noise_std_m: {args.noise_std}")
    print(f"start_x_m: {generated.trajectory_x_m[0]:.3f}")
    print(f"start_y_m: {generated.trajectory_y_m[0]:.3f}")


if __name__ == "__main__":
    main()
