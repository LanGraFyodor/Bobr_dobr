from __future__ import annotations

import ctypes
import platform
import subprocess
from pathlib import Path

import numpy as np
from affine import Affine


ROOT_DIR = Path(__file__).resolve().parents[3]
RUST_CRATE_DIR = ROOT_DIR / "rust" / "terrain_nav_core"


class RustCoreUnavailable(RuntimeError):
    pass


class _RustCandidate(ctypes.Structure):
    _fields_ = [
        ("start_x_m", ctypes.c_double),
        ("start_y_m", ctypes.c_double),
        ("speed_mps", ctypes.c_double),
        ("azimuth_deg", ctypes.c_double),
        ("error_rmse_m", ctypes.c_double),
        ("correlation", ctypes.c_double),
        ("speed_index", ctypes.c_size_t),
        ("azimuth_index", ctypes.c_size_t),
    ]


_LIB: ctypes.CDLL | None = None


def search_candidates_rust(
    dem: np.ndarray,
    transform: Affine,
    measured_profile_m: np.ndarray,
    timestamps_s: np.ndarray,
    start_points_m: np.ndarray,
    speeds_mps: np.ndarray,
    azimuths_deg: np.ndarray,
    top_k: int,
    use_weighted_scoring: bool = True,
) -> list[dict[str, float | int]]:
    lib = _load_library()
    dem_arr = _as_float64_contiguous(dem)
    measured = _as_float64_contiguous(measured_profile_m)
    timestamps = _as_float64_contiguous(timestamps_s)
    starts = _as_float64_contiguous(start_points_m).reshape(-1, 2)
    speeds = _as_float64_contiguous(speeds_mps)
    azimuths = _as_float64_contiguous(azimuths_deg)
    top_k = max(1, int(top_k))

    out_candidates = (_RustCandidate * top_k)()
    out_count = ctypes.c_size_t(0)

    status = lib.terrain_nav_search(
        _ptr(dem_arr),
        ctypes.c_size_t(dem_arr.shape[0]),
        ctypes.c_size_t(dem_arr.shape[1]),
        ctypes.c_double(transform.c),
        ctypes.c_double(transform.f),
        ctypes.c_double(transform.a),
        ctypes.c_double(transform.e),
        _ptr(measured),
        _ptr(timestamps),
        ctypes.c_size_t(measured.size),
        _ptr(starts),
        ctypes.c_size_t(starts.shape[0]),
        _ptr(speeds),
        ctypes.c_size_t(speeds.size),
        _ptr(azimuths),
        ctypes.c_size_t(azimuths.size),
        ctypes.c_int(1 if use_weighted_scoring else 0),
        ctypes.c_size_t(top_k),
        out_candidates,
        ctypes.byref(out_count),
    )
    if status < 0:
        raise RustCoreUnavailable("Rust search received invalid input")
    if status > 0 or out_count.value == 0:
        return []

    return [
        {
            "start_x": float(candidate.start_x_m),
            "start_y": float(candidate.start_y_m),
            "speed": float(candidate.speed_mps),
            "azimuth": float(candidate.azimuth_deg),
            "error": float(candidate.error_rmse_m),
            "correlation": float(candidate.correlation),
            "speed_index": int(candidate.speed_index),
            "azimuth_index": int(candidate.azimuth_index),
        }
        for candidate in out_candidates[: out_count.value]
    ]


def compute_error_grid_rust(
    dem: np.ndarray,
    transform: Affine,
    measured_profile_m: np.ndarray,
    timestamps_s: np.ndarray,
    start_x_m: float,
    start_y_m: float,
    speeds_mps: np.ndarray,
    azimuths_deg: np.ndarray,
    use_weighted_scoring: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    lib = _load_library()
    dem_arr = _as_float64_contiguous(dem)
    measured = _as_float64_contiguous(measured_profile_m)
    timestamps = _as_float64_contiguous(timestamps_s)
    speeds = _as_float64_contiguous(speeds_mps)
    azimuths = _as_float64_contiguous(azimuths_deg)
    errors = np.empty((speeds.size, azimuths.size), dtype=np.float64)
    correlations = np.empty_like(errors)

    status = lib.terrain_nav_error_grid(
        _ptr(dem_arr),
        ctypes.c_size_t(dem_arr.shape[0]),
        ctypes.c_size_t(dem_arr.shape[1]),
        ctypes.c_double(transform.c),
        ctypes.c_double(transform.f),
        ctypes.c_double(transform.a),
        ctypes.c_double(transform.e),
        _ptr(measured),
        _ptr(timestamps),
        ctypes.c_size_t(measured.size),
        ctypes.c_double(float(start_x_m)),
        ctypes.c_double(float(start_y_m)),
        _ptr(speeds),
        ctypes.c_size_t(speeds.size),
        _ptr(azimuths),
        ctypes.c_size_t(azimuths.size),
        ctypes.c_int(1 if use_weighted_scoring else 0),
        _ptr(errors),
        _ptr(correlations),
    )
    if status != 0:
        raise RustCoreUnavailable("Rust error grid received invalid input")

    scores = compute_match_scores(errors, correlations)
    return errors, correlations, int(np.argmin(scores))


def compute_match_scores(
    errors: np.ndarray,
    correlations: np.ndarray,
    correlation_weight: float = 0.35,
) -> np.ndarray:
    scores = np.full(errors.shape, np.inf, dtype=np.float64)
    finite_errors = np.isfinite(errors)
    safe_correlations = np.where(np.isfinite(correlations), correlations, -1.0)
    penalty = 1.0 + float(correlation_weight) * np.maximum(0.0, 1.0 - np.clip(safe_correlations, -1.0, 1.0))
    scores[finite_errors] = errors[finite_errors] * penalty[finite_errors]
    return scores


def _load_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    library_path = _library_path()
    if not library_path.exists():
        _build_library()
    if not library_path.exists():
        raise RustCoreUnavailable(f"Rust library was not built: {library_path}")

    lib = ctypes.CDLL(str(library_path))
    _configure_signatures(lib)
    _LIB = lib
    return lib


def _build_library() -> None:
    if not RUST_CRATE_DIR.exists():
        raise RustCoreUnavailable(f"Rust crate is missing: {RUST_CRATE_DIR}")

    try:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=RUST_CRATE_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RustCoreUnavailable(f"Failed to build Rust core: {exc}") from exc


def _library_path() -> Path:
    release_dir = RUST_CRATE_DIR / "target" / "release"
    system = platform.system().lower()
    if system == "windows":
        return release_dir / "terrain_nav_core.dll"
    if system == "darwin":
        return release_dir / "libterrain_nav_core.dylib"
    return release_dir / "libterrain_nav_core.so"


def _configure_signatures(lib: ctypes.CDLL) -> None:
    lib.terrain_nav_search.restype = ctypes.c_int
    lib.terrain_nav_error_grid.restype = ctypes.c_int


def _as_float64_contiguous(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def _ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
