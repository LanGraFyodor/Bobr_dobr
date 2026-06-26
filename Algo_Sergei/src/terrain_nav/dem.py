from __future__ import annotations

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import xy

from terrain_nav.models import DemData


def read_dem_as_utm(path: str, band: int = 1) -> DemData:
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError("DEM does not contain CRS metadata")

        heights = dataset.read(band)
        source_crs = CRS.from_user_input(dataset.crs)
        nodata = dataset.nodata

        lon, lat = raster_pixel_lonlat_grids(dataset)
        utm_crs = utm_crs_for_lonlat(float(np.nanmean(lon)), float(np.nanmean(lat)))
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        x_utm, y_utm = transformer.transform(lon, lat)

    return DemData(
        heights=heights,
        x_utm=np.asarray(x_utm, dtype=np.float64),
        y_utm=np.asarray(y_utm, dtype=np.float64),
        source_crs=source_crs,
        utm_crs=utm_crs,
        nodata=nodata,
    )


def raster_pixel_lonlat_grids(dataset: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((dataset.height, dataset.width))
    xs, ys = xy(dataset.transform, rows, cols, offset="center")

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    source_crs = CRS.from_user_input(dataset.crs)
    if source_crs.is_geographic:
        return xs, ys

    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xs, ys)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def dataset_center_lonlat(dataset: rasterio.io.DatasetReader, source_crs: CRS) -> tuple[float, float]:
    center_x = (dataset.bounds.left + dataset.bounds.right) * 0.5
    center_y = (dataset.bounds.bottom + dataset.bounds.top) * 0.5

    if source_crs.is_geographic:
        return float(center_x), float(center_y)

    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(center_x, center_y)
    return float(lon), float(lat)


def dataset_center_utm(
    dataset: rasterio.io.DatasetReader,
    source_crs: CRS,
    utm_crs: CRS,
) -> tuple[float, float]:
    lon, lat = dataset_center_lonlat(dataset, source_crs)
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def dataset_utm_bounds(
    dataset: rasterio.io.DatasetReader,
    source_crs: CRS,
    utm_crs: CRS,
) -> tuple[float, float, float, float]:
    xs = np.array(
        [dataset.bounds.left, dataset.bounds.left, dataset.bounds.right, dataset.bounds.right],
        dtype=np.float64,
    )
    ys = np.array(
        [dataset.bounds.bottom, dataset.bounds.top, dataset.bounds.bottom, dataset.bounds.top],
        dtype=np.float64,
    )

    if source_crs != utm_crs:
        transformer = Transformer.from_crs(source_crs, utm_crs, always_xy=True)
        xs, ys = transformer.transform(xs, ys)

    return float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys))


def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude is outside valid range: {lon}")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude is outside valid range: {lat}")

    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    epsg = 32600 + zone if lat >= 0.0 else 32700 + zone
    return CRS.from_epsg(epsg)


# Backward-compatible aliases for older imports.
_utm_crs_for_lonlat = utm_crs_for_lonlat
_dataset_center_lonlat = dataset_center_lonlat
_dataset_center_utm = dataset_center_utm
