from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from terrain_nav.core.flat_gap import FlatGap, FlatGapBridge, bridge_flat_gap, detect_flat_gap
from terrain_nav.core.geometry import compute_trajectory_points
from terrain_nav.core.sampling import sample_utm_raster_heights
from terrain_nav.core.search import localize_position_from_nmea
from terrain_nav.core.utm_raster import load_utm_raster
from terrain_nav.io.nmea import parse_nmea_profile
from terrain_nav.models import LocalizationResult
from terrain_nav.simulation.generator import make_gga_sentence


@dataclass(frozen=True)
class FlatGapBridgeResult:
    line: LocalizationResult
    before: LocalizationResult
    after: LocalizationResult
    gap: FlatGap
    bridge: FlatGapBridge
    trajectory_x_m: np.ndarray
    trajectory_y_m: np.ndarray
    before_end_index: int
    after_start_index: int
    before_point_count: int
    after_point_count: int
    after_anchor_cross_track_error_m: float
    after_anchor_along_track_error_m: float
    after_validation_rmse_m: float
    after_validation_correlation: float
    confidence: float
    estimated_accuracy_m: float
    quality_label: str
    is_reliable: bool
    rejection_reason: str | None
    has_flat_gap: bool

    @property
    def speed_mps(self) -> float:
        return self.line.speed_mps

    @property
    def azimuth_deg(self) -> float:
        return self.line.azimuth_deg

    @property
    def velocity_x_mps(self) -> float:
        return self.line.velocity_x_mps

    @property
    def velocity_y_mps(self) -> float:
        return self.line.velocity_y_mps

    @property
    def start_x_m(self) -> float:
        return self.line.start_x_m

    @property
    def start_y_m(self) -> float:
        return self.line.start_y_m

    @property
    def current_x_m(self) -> float:
        return float(self.trajectory_x_m[-1])

    @property
    def current_y_m(self) -> float:
        return float(self.trajectory_y_m[-1])


def localize_with_flat_gap_bridge(
    dem_path: str | Path,
    nmea_path: str | Path,
    *,
    baro_altitude_m: float = 1500.0,
    sample_rate_hz: float = 1.0,
    min_segment_points: int = 40,
    flat_window_points: int = 31,
    gap_variance_threshold_m2: float = 25.0,
    gap_gradient_threshold_m: float = 8.0,
    min_flat_duration_s: float = 60.0,
    bridge_decay_time_s: float = 180.0,
    max_before_candidates: int = 4,
    min_gap_start_fraction: float = 0.45,
    **localize_kwargs,
) -> FlatGapBridgeResult:
    localize_kwargs = dict(localize_kwargs)
    localize_kwargs.pop("auto_retry_unweighted", None)

    profile = parse_nmea_profile(nmea_path, baro_altitude_m, sample_rate_hz)
    gap = detect_flat_gap(
        profile.terrain_profile_m,
        profile.timestamps_s,
        window_points=flat_window_points,
        variance_threshold_m2=gap_variance_threshold_m2,
        gradient_threshold_m=gap_gradient_threshold_m,
        min_duration_s=min_flat_duration_s,
        min_start_index=int(profile.timestamps_s.size * float(np.clip(min_gap_start_fraction, 0.0, 0.9))),
    )
    if gap is None:
        return _localize_straight_hypothesis(
            dem_path=dem_path,
            nmea_path=nmea_path,
            profile=profile,
            baro_altitude_m=baro_altitude_m,
            sample_rate_hz=sample_rate_hz,
            localize_kwargs=localize_kwargs,
        )

    before_stop = gap.start_index
    after_start = gap.end_index + 1
    if before_stop < min_segment_points:
        return _localize_straight_hypothesis(
            dem_path=dem_path,
            nmea_path=nmea_path,
            profile=profile,
            baro_altitude_m=baro_altitude_m,
            sample_rate_hz=sample_rate_hz,
            localize_kwargs=localize_kwargs,
        )
    after_point_count = max(0, int(profile.timestamps_s.size - after_start))
    has_after_validation = after_point_count >= min_segment_points

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        before_path = tmp / "before_flat_gap_profile.nmea"
        _write_profile_nmea(
            before_path,
            profile.timestamps_s[:before_stop],
            profile.radio_altitudes_m[:before_stop],
        )

        before = localize_position_from_nmea(
            dem_path,
            before_path,
            baro_altitude_m=baro_altitude_m,
            sample_rate_hz=sample_rate_hz,
            auto_retry_unweighted=False,
            **localize_kwargs,
        )

    full_x, full_y = compute_trajectory_points(
        start_x_m=before.start_x_m,
        start_y_m=before.start_y_m,
        speed_mps=before.speed_mps,
        azimuth_deg=before.azimuth_deg,
        timestamps_s=profile.timestamps_s,
    )
    entry_x = float(full_x[gap.start_index])
    entry_y = float(full_y[gap.start_index])

    full_predicted_profile = _sample_trajectory(dem_path, full_x, full_y)
    if has_after_validation:
        after_rmse, after_correlation = _profile_match_metrics(
            profile.terrain_profile_m[after_start:],
            full_predicted_profile[after_start:],
        )
        after_confidence = _validation_confidence(after_rmse, after_correlation)
        cross_track_error, along_track_error = _anchor_errors_against_line(
            line_start_x_m=before.start_x_m,
            line_start_y_m=before.start_y_m,
            azimuth_deg=before.azimuth_deg,
            anchor_x_m=float(full_x[after_start]),
            anchor_y_m=float(full_y[after_start]),
            expected_x_m=float(full_x[after_start]),
            expected_y_m=float(full_y[after_start]),
        )
    else:
        after_rmse = float("nan")
        after_correlation = float("nan")
        after_confidence = before.confidence
        cross_track_error = 0.0
        along_track_error = 0.0
    bridge = bridge_flat_gap(
        timestamps_s=profile.timestamps_s,
        gap=gap,
        entry_x_m=entry_x,
        entry_y_m=entry_y,
        speed_mps=before.speed_mps,
        azimuth_deg=before.azimuth_deg,
        entry_confidence=before.confidence,
        decay_time_s=bridge_decay_time_s,
    )

    confidence = float(min(before.confidence, after_confidence, float(bridge.confidence[-1])))
    if not has_after_validation:
        confidence *= 0.75
    anchor_disagreement = float(abs(cross_track_error) + 0.25 * abs(along_track_error))
    after_penalty = 0.0 if not has_after_validation else max(after_rmse if np.isfinite(after_rmse) else 1_000.0, 0.0) * 0.50
    estimated_accuracy = float(
        before.estimated_accuracy_m
        + after_penalty
        + anchor_disagreement * 0.25
        + max(0.0, 1.0 - confidence) * 100.0
    )
    quality = _quality_label(estimated_accuracy, confidence)
    is_reliable, rejection_reason = _bridge_reliability(
        before_confidence=before.confidence,
        after_rmse_m=after_rmse,
        after_correlation=after_correlation,
        bridge_confidence=confidence,
        estimated_accuracy_m=estimated_accuracy,
        has_after_validation=has_after_validation,
    )

    return FlatGapBridgeResult(
        line=before,
        before=before,
        after=before,
        gap=gap,
        bridge=bridge,
        trajectory_x_m=full_x,
        trajectory_y_m=full_y,
        before_end_index=before_stop - 1,
        after_start_index=after_start,
        before_point_count=int(before_stop),
        after_point_count=after_point_count,
        after_anchor_cross_track_error_m=cross_track_error,
        after_anchor_along_track_error_m=along_track_error,
        after_validation_rmse_m=after_rmse,
        after_validation_correlation=after_correlation,
        confidence=confidence,
        estimated_accuracy_m=estimated_accuracy,
        quality_label=quality,
        is_reliable=is_reliable,
        rejection_reason=rejection_reason,
        has_flat_gap=True,
    )


def _localize_straight_hypothesis(
    dem_path: str | Path,
    nmea_path: str | Path,
    profile,
    baro_altitude_m: float,
    sample_rate_hz: float,
    localize_kwargs: dict,
) -> FlatGapBridgeResult:
    line = localize_position_from_nmea(
        dem_path,
        nmea_path,
        baro_altitude_m=baro_altitude_m,
        sample_rate_hz=sample_rate_hz,
        auto_retry_unweighted=False,
        **localize_kwargs,
    )
    full_x, full_y = compute_trajectory_points(
        start_x_m=line.start_x_m,
        start_y_m=line.start_y_m,
        speed_mps=line.speed_mps,
        azimuth_deg=line.azimuth_deg,
        timestamps_s=profile.timestamps_s,
    )
    fake_gap_index = max(0, profile.timestamps_s.size - 1)
    fake_gap = FlatGap(
        start_index=fake_gap_index,
        end_index=fake_gap_index,
        start_time_s=float(profile.timestamps_s[fake_gap_index]),
        end_time_s=float(profile.timestamps_s[fake_gap_index]),
        duration_s=0.0,
        variance_m2=float(np.nanvar(profile.terrain_profile_m)),
    )
    bridge = bridge_flat_gap(
        timestamps_s=profile.timestamps_s,
        gap=fake_gap,
        entry_x_m=float(full_x[fake_gap_index]),
        entry_y_m=float(full_y[fake_gap_index]),
        speed_mps=line.speed_mps,
        azimuth_deg=line.azimuth_deg,
        entry_confidence=line.confidence,
    )
    reliable = bool(line.confidence >= 0.25 and (not np.isfinite(line.estimated_accuracy_m) or line.estimated_accuracy_m <= 250.0))
    reason = None if reliable else "весь профиль обработан, но геопривязка неоднозначна"
    return FlatGapBridgeResult(
        line=line,
        before=line,
        after=line,
        gap=fake_gap,
        bridge=bridge,
        trajectory_x_m=full_x,
        trajectory_y_m=full_y,
        before_end_index=profile.timestamps_s.size - 1,
        after_start_index=profile.timestamps_s.size,
        before_point_count=int(profile.timestamps_s.size),
        after_point_count=0,
        after_anchor_cross_track_error_m=0.0,
        after_anchor_along_track_error_m=0.0,
        after_validation_rmse_m=float(line.best_error),
        after_validation_correlation=float(line.best_correlation),
        confidence=float(line.confidence),
        estimated_accuracy_m=float(line.estimated_accuracy_m),
        quality_label=line.quality_label,
        is_reliable=reliable,
        rejection_reason=reason,
        has_flat_gap=False,
    )


def _find_best_before_anchor(
    dem_path: str | Path,
    tmp_dir: Path,
    profile,
    before_stop: int,
    min_segment_points: int,
    baro_altitude_m: float,
    sample_rate_hz: float,
    max_candidates: int,
    localize_kwargs: dict,
) -> tuple[LocalizationResult, int]:
    candidate_stops = _make_before_candidate_stops(
        before_stop=before_stop,
        profile_size=profile.timestamps_s.size,
        min_segment_points=min_segment_points,
        max_candidates=max_candidates,
    )

    best_result: LocalizationResult | None = None
    best_stop = candidate_stops[0]
    best_score = -float("inf")

    for index, stop in enumerate(candidate_stops):
        before_path = tmp_dir / f"before_gap_{index}.nmea"
        _write_profile_nmea(
            before_path,
            profile.timestamps_s[:stop],
            profile.radio_altitudes_m[:stop],
        )
        result = localize_position_from_nmea(
            dem_path,
            before_path,
            baro_altitude_m=baro_altitude_m,
            sample_rate_hz=sample_rate_hz,
            auto_retry_unweighted=False,
            **localize_kwargs,
        )
        score = _before_anchor_score(result, stop)
        if best_result is None or score > best_score:
            best_result = result
            best_stop = stop
            best_score = score

    if best_result is None:
        raise ValueError("Не удалось найти надежный исторический anchor перед плоскостью.")
    return best_result, best_stop


def _make_before_candidate_stops(
    before_stop: int,
    profile_size: int,
    min_segment_points: int,
    max_candidates: int,
) -> list[int]:
    max_candidates = max(1, int(max_candidates))
    guard = max(20, min(90, profile_size // 12))
    raw = [
        before_stop,
        before_stop - guard,
        before_stop - guard * 2,
        before_stop - guard * 3,
    ]
    stops: list[int] = []
    for stop in raw:
        stop = int(np.clip(stop, min_segment_points, before_stop))
        if stop >= min_segment_points and stop not in stops:
            stops.append(stop)
        if len(stops) >= max_candidates:
            break
    return stops or [before_stop]


def _before_anchor_score(result: LocalizationResult, point_count: int) -> float:
    accuracy = result.estimated_accuracy_m if np.isfinite(result.estimated_accuracy_m) else 1e9
    error = result.best_error if np.isfinite(result.best_error) else 1e9
    length_bonus = min(float(point_count), 500.0) * 0.05
    relief_bonus = min(float(np.sqrt(max(result.terrain_variance_m2, 0.0))), 200.0) * 0.2
    return (
        float(result.confidence) * 1000.0
        + length_bonus
        + relief_bonus
        - float(accuracy) * 0.10
        - float(error) * 0.50
    )


def _write_profile_nmea(path: Path, timestamps_s: np.ndarray, radio_altitudes_m: np.ndarray) -> None:
    lines = [make_gga_sentence(float(t), float(h)) for t, h in zip(timestamps_s, radio_altitudes_m)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_trajectory(
    dem_path: str | Path,
    trajectory_x_m: np.ndarray,
    trajectory_y_m: np.ndarray,
) -> np.ndarray:
    import rasterio

    with rasterio.open(dem_path) as dataset:
        raster = load_utm_raster(dataset)
    return sample_utm_raster_heights(raster.heights, raster.transform, trajectory_x_m, trajectory_y_m)


def _profile_match_metrics(measured_m: np.ndarray, predicted_m: np.ndarray) -> tuple[float, float]:
    measured = np.asarray(measured_m, dtype=np.float64)
    predicted = np.asarray(predicted_m, dtype=np.float64)
    mask = np.isfinite(measured) & np.isfinite(predicted)
    if int(np.count_nonzero(mask)) < 3:
        return float("inf"), -1.0

    measured = measured[mask]
    predicted = predicted[mask]
    rmse = float(np.sqrt(np.mean((measured - predicted) ** 2)))
    measured_centered = measured - float(np.mean(measured))
    predicted_centered = predicted - float(np.mean(predicted))
    denom = float(np.linalg.norm(measured_centered) * np.linalg.norm(predicted_centered))
    if denom <= 1e-9:
        correlation = 1.0 if rmse <= 5.0 else 0.0
    else:
        correlation = float(np.dot(measured_centered, predicted_centered) / denom)
    return rmse, float(np.clip(correlation, -1.0, 1.0))


def _validation_confidence(rmse_m: float, correlation: float) -> float:
    if not np.isfinite(rmse_m):
        return 0.0
    error_score = float(np.exp(-max(rmse_m, 0.0) / 80.0))
    correlation_score = float(np.clip((correlation + 1.0) * 0.5, 0.0, 1.0))
    return float(np.clip(error_score * correlation_score, 0.0, 1.0))


def _bridge_reliability(
    before_confidence: float,
    after_rmse_m: float,
    after_correlation: float,
    bridge_confidence: float,
    estimated_accuracy_m: float,
    has_after_validation: bool = True,
) -> tuple[bool, str | None]:
    if before_confidence < 0.35:
        return False, "информативный участок до плоскости не дал уверенной геопривязки"
    if not has_after_validation:
        return False, "после слабого участка мало данных для независимой проверки продолжения"
    if not np.isfinite(after_rmse_m) or after_rmse_m > 80.0:
        return False, "участок после плоскости не подтверждает продолжение прямой"
    if after_correlation < 0.35:
        return False, "форма профиля после плоскости слабо похожа на DEM по прямой"
    if bridge_confidence < 0.25:
        return False, "накопленная неопределенность моста слишком высокая"
    if np.isfinite(estimated_accuracy_m) and estimated_accuracy_m > 250.0:
        return False, "оценка точности слишком грубая для выдачи координат"
    return True, None


def _quality_label(estimated_accuracy_m: float, confidence: float) -> str:
    if estimated_accuracy_m <= 80.0 and confidence >= 0.6:
        return "высокая"
    if estimated_accuracy_m <= 200.0 and confidence >= 0.3:
        return "средняя"
    return "низкая"


def _anchor_errors_against_line(
    line_start_x_m: float,
    line_start_y_m: float,
    azimuth_deg: float,
    anchor_x_m: float,
    anchor_y_m: float,
    expected_x_m: float,
    expected_y_m: float,
) -> tuple[float, float]:
    azimuth_rad = np.deg2rad(azimuth_deg)
    along_x = float(np.sin(azimuth_rad))
    along_y = float(np.cos(azimuth_rad))
    normal_x = along_y
    normal_y = -along_x

    dx = float(anchor_x_m) - float(line_start_x_m)
    dy = float(anchor_y_m) - float(line_start_y_m)
    cross_track = dx * normal_x + dy * normal_y

    expected_along = (float(expected_x_m) - float(line_start_x_m)) * along_x + (
        float(expected_y_m) - float(line_start_y_m)
    ) * along_y
    anchor_along = dx * along_x + dy * along_y
    along_error = anchor_along - expected_along
    return float(cross_track), float(along_error)
