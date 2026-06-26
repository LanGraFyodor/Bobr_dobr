from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import xy


@dataclass(frozen=True)
class DemData:
    """DEM heights and matching projected coordinate grids."""

    heights: np.ndarray
    x_utm: np.ndarray
    y_utm: np.ndarray
    source_crs: CRS
    utm_crs: CRS
    nodata: float | int | None


def read_dem_as_utm(path: str, band: int = 1) -> DemData:
    """Read a DEM GeoTIFF and return heights plus UTM coordinates in meters.

    The returned arrays all have shape ``(height, width)``:
    - ``heights`` contains DEM elevations from the selected raster band.
    - ``x_utm`` contains UTM easting for each pixel center.
    - ``y_utm`` contains UTM northing for each pixel center.

    If the source map is already projected, coordinates are first converted to
    WGS84 lon/lat and then to the UTM zone selected by the raster center.
    """

    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError("DEM does not contain CRS metadata")

        heights = dataset.read(band)
        source_crs = CRS.from_user_input(dataset.crs)
        nodata = dataset.nodata

        lon, lat = _raster_pixel_lonlat_grids(dataset)
        utm_crs = _utm_crs_for_lonlat(float(np.nanmean(lon)), float(np.nanmean(lat)))

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


def _raster_pixel_lonlat_grids(dataset: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
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


def _utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude is outside valid range: {lon}")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude is outside valid range: {lat}")

    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)

    epsg = 32600 + zone if lat >= 0.0 else 32700 + zone
    return CRS.from_epsg(epsg)


if __name__ == "__main__":
    dem = read_dem_as_utm("map.tif")
    print(f"heights shape: {dem.heights.shape}")
    print(f"source CRS: {dem.source_crs.to_string()}")
    print(f"UTM CRS: {dem.utm_crs.to_string()}")
    print(f"x range: {np.nanmin(dem.x_utm):.2f} .. {np.nanmax(dem.x_utm):.2f} m")
    print(f"y range: {np.nanmin(dem.y_utm):.2f} .. {np.nanmax(dem.y_utm):.2f} m")
