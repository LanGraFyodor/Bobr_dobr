from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(nogil=True, parallel=True)
def _compute_errors_numba(
    dem: np.ndarray,
    nodata: float,
    measured_profile_m: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    shape_speeds: int,
    shape_azimuths: int,
    shape_points: int,
) -> np.ndarray:
    """
    Computes RMSE for a grid of trajectories using bilinear interpolation.
    rows and cols are flat arrays of shape (shape_speeds * shape_azimuths * shape_points).
    """
    # Initialize output array of shape (shape_speeds, shape_azimuths)
    errors = np.full((shape_speeds, shape_azimuths), np.inf, dtype=np.float64)
    
    height, width = dem.shape
    
    # Parallel loop over combinations of speed and azimuth
    total_combinations = shape_speeds * shape_azimuths
    
    for i in prange(total_combinations):
        speed_idx = i // shape_azimuths
        azimuth_idx = i % shape_azimuths
        
        start_idx = i * shape_points
        
        squared_error_sum = 0.0
        valid_count = 0
        
        for p in range(shape_points):
            idx = start_idx + p
            r = rows[idx]
            c = cols[idx]
            
            # Check bounds
            if not np.isfinite(r) or not np.isfinite(c) or r < 0 or c < 0 or r >= height - 1 or c >= width - 1:
                continue
                
            # Bilinear interpolation
            r0 = int(np.floor(r))
            c0 = int(np.floor(c))
            r1 = r0 + 1
            c1 = c0 + 1
            
            # Check nodata
            v00 = dem[r0, c0]
            v01 = dem[r0, c1]
            v10 = dem[r1, c0]
            v11 = dem[r1, c1]
            
            if v00 == nodata or v01 == nodata or v10 == nodata or v11 == nodata:
                continue
                
            dr = r - r0
            dc = c - c0
            
            interp = (
                v00 * (1 - dr) * (1 - dc) +
                v01 * (1 - dr) * dc +
                v10 * dr * (1 - dc) +
                v11 * dr * dc
            )
            
            diff = interp - measured_profile_m[p]
            squared_error_sum += diff * diff
            valid_count += 1
            
        if valid_count == shape_points:
            errors[speed_idx, azimuth_idx] = np.sqrt(squared_error_sum / shape_points)
            
    return errors
