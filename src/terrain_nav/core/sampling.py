from __future__ import annotations

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.transform import rowcol
from scipy.ndimage import map_coordinates


def sample_dem_heights(
    dem: np.ndarray,
    dataset: rasterio.io.DatasetReader,
    source_crs: CRS,
    utm_crs: CRS,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> np.ndarray:
    x_source, y_source = utm_to_source_xy(source_crs, utm_crs, x_m, y_m)
    rows, cols = rowcol(dataset.transform, x_source, y_source, op=np.float64)

    coords = np.vstack([np.asarray(rows, dtype=np.float64) - 0.5, np.asarray(cols, dtype=np.float64) - 0.5])
    sampled = map_coordinates(dem, coords, order=1, mode="constant", cval=np.nan)
    return np.asarray(sampled, dtype=np.float64)


def sample_utm_raster_heights(
    dem: np.ndarray,
    transform: Affine,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> np.ndarray:
    if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9:
        raise ValueError("Rotated rasters are not supported")

    cols = (np.asarray(x_m, dtype=np.float64) - transform.c) / transform.a - 0.5
    rows = (np.asarray(y_m, dtype=np.float64) - transform.f) / transform.e - 0.5
    coords = np.vstack([rows, cols])
    sampled = map_coordinates(dem, coords, order=1, mode="constant", cval=np.nan)
    return np.asarray(sampled, dtype=np.float64)


def utm_to_source_xy(
    source_crs: CRS,
    utm_crs: CRS,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if source_crs == utm_crs:
        return x_m, y_m

    transformer = Transformer.from_crs(utm_crs, source_crs, always_xy=True)
    x_source, y_source = transformer.transform(x_m, y_m)
    return np.asarray(x_source, dtype=np.float64), np.asarray(y_source, dtype=np.float64)


_utm_to_source_xy = utm_to_source_xy
