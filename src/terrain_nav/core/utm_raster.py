from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

from terrain_nav.io.dem import dataset_center_lonlat, dataset_center_utm, utm_crs_for_lonlat


@dataclass(frozen=True)
class UtmRaster:
    heights: np.ndarray
    transform: Affine
    crs: CRS
    bounds: tuple[float, float, float, float]
    center: tuple[float, float]


def load_utm_raster(dataset: rasterio.io.DatasetReader) -> UtmRaster:
    if dataset.crs is None:
        raise ValueError("DEM does not contain CRS metadata")

    source_crs = CRS.from_user_input(dataset.crs)
    center_lon, center_lat = dataset_center_lonlat(dataset, source_crs)
    utm_crs = utm_crs_for_lonlat(center_lon, center_lat)

    if source_crs == utm_crs:
        heights = dataset.read(1).astype(np.float64)
        transform = dataset.transform
    else:
        transform, width, height = calculate_default_transform(
            source_crs,
            utm_crs,
            dataset.width,
            dataset.height,
            *dataset.bounds,
        )
        heights = np.full((height, width), np.nan, dtype=np.float64)
        reproject(
            source=rasterio.band(dataset, 1),
            destination=heights,
            src_transform=dataset.transform,
            src_crs=source_crs,
            src_nodata=dataset.nodata,
            dst_transform=transform,
            dst_crs=utm_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    if dataset.nodata is not None:
        heights[heights == dataset.nodata] = np.nan

    if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9:
        raise ValueError("UTM DEM transform contains rotation; only north-up rasters are supported")

    bounds = raster_bounds(transform, heights.shape[1], heights.shape[0])
    center = dataset_center_utm(dataset, source_crs, utm_crs)
    return UtmRaster(
        heights=np.ascontiguousarray(heights, dtype=np.float64),
        transform=transform,
        crs=utm_crs,
        bounds=bounds,
        center=(float(center[0]), float(center[1])),
    )


def raster_bounds(transform: Affine, width: int, height: int) -> tuple[float, float, float, float]:
    left = transform.c
    right = transform.c + transform.a * width
    top = transform.f
    bottom = transform.f + transform.e * height
    min_x, max_x = sorted((float(left), float(right)))
    min_y, max_y = sorted((float(bottom), float(top)))
    return min_x, max_x, min_y, max_y
