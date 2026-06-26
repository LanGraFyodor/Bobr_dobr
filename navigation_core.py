from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from terrain_nav.geometry import compute_trajectory_points, smooth_trajectory
from terrain_nav.models import LocalizationResult, NmeaProfile, SearchGrid
from terrain_nav.nmea import parse_nmea_profile
from terrain_nav.sampling import sample_dem_heights
from terrain_nav.search import (
    compute_confidence,
    localize_from_nmea,
    localize_position_from_nmea,
    make_search_grid,
)

__all__ = [
    "LocalizationResult",
    "NmeaProfile",
    "SearchGrid",
    "compute_confidence",
    "compute_trajectory_points",
    "localize_from_nmea",
    "localize_position_from_nmea",
    "make_search_grid",
    "parse_nmea_profile",
    "sample_dem_heights",
    "smooth_trajectory",
]
