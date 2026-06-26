from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from terrain_nav.dem import (
    DemData,
    dataset_center_lonlat,
    dataset_center_utm,
    read_dem_as_utm,
    utm_crs_for_lonlat,
)

_dataset_center_lonlat = dataset_center_lonlat
_dataset_center_utm = dataset_center_utm
_utm_crs_for_lonlat = utm_crs_for_lonlat

__all__ = [
    "DemData",
    "read_dem_as_utm",
    "dataset_center_lonlat",
    "dataset_center_utm",
    "utm_crs_for_lonlat",
]
