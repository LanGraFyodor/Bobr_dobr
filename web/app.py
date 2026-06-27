from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from pyproj import CRS, Transformer
from rasterio.transform import rowcol

from terrain_nav.io.dem import dataset_center_lonlat, utm_crs_for_lonlat
from terrain_nav.core.search import localize_position_from_nmea
from terrain_nav.simulation import generate_test_flight


DEFAULT_DEM_PATH = ROOT_DIR / "data" / "map.tif"
DEFAULT_NMEA_PATH = Path("outputs/test_flight.nmea")


def main() -> None:
    st.set_page_config(page_title="Полет вслепую", layout="wide")

    st.title("Полет вслепую")
    st.caption("Определение скорости и курса по DEM и данным радиовысотомера")
    st.info(
        "Быстрый сценарий: 1) оставьте `map.tif` или загрузите свою карту; "
        "2) нажмите `Сгенерировать NMEA` в боковой панели или загрузите NMEA; "
        "3) нажмите `Запустить локализацию`."
    )

    with st.sidebar:
        st.header("Данные")
        dem_path = _select_dem_path()
        nmea_path = _select_nmea_path(dem_path)

        st.header("Поиск")
        sample_rate_hz = st.number_input("Частота NMEA, Гц", min_value=0.1, max_value=10.0, value=1.0, step=0.5)
        baro_altitude_m = st.number_input("Барометрическая высота, м", min_value=100.0, value=1500.0, step=50.0)
        min_speed_mps = st.number_input("Мин. скорость, м/с", min_value=1.0, value=10.0, step=1.0)
        max_speed_mps = st.number_input("Макс. скорость, м/с", min_value=1.0, value=50.0, step=1.0)
        coarse_speed_step_mps = st.number_input("Грубый шаг скорости, м/с", min_value=1.0, value=5.0, step=1.0)
        fine_speed_step_mps = st.number_input("Точный шаг скорости, м/с", min_value=0.5, value=1.0, step=0.5)
        coarse_azimuth_step_deg = st.number_input("Грубый шаг азимута, град", min_value=1.0, max_value=45.0, value=5.0, step=1.0)
        fine_azimuth_step_deg = st.number_input("Точный шаг азимута, град", min_value=1.0, max_value=10.0, value=1.0, step=1.0)
        coarse_start_step_m = st.number_input("Грубый шаг X/Y, м", min_value=1000.0, value=5000.0, step=1000.0)
        refine_radius_m = st.number_input("Радиус уточнения X/Y, м", min_value=0.0, value=10000.0, step=500.0)
        refine_start_step_m = st.number_input("Точный шаг X/Y, м", min_value=100.0, value=1000.0, step=100.0)
        coarse_top_k = st.number_input("Кандидатов для уточнения", min_value=1, max_value=100, value=50, step=1)
        coarse_profile_points = st.number_input("Точек для грубого поиска", min_value=50, max_value=1000, value=250, step=10)
        max_profile_points = st.number_input("Точек профиля для поиска", min_value=50, max_value=5000, value=600, step=50)
        search_radius_m = st.number_input("Радиус поиска от центра, м (0 = вся карта)", min_value=0.0, value=0.0, step=5000.0)
        flat_variance_threshold_m2 = st.number_input("Порог плоского рельефа, м²", min_value=0.0, value=1.0, step=0.5)
        use_weighted_scoring = st.checkbox("Weighted scoring для плоских участков", value=False)
        smoothing_method_label = st.selectbox("Сглаживание траектории", ["Фильтр Калмана", "Скользящее среднее"])
        smoothing_method = "kalman" if smoothing_method_label == "Фильтр Калмана" else "moving_average"
        smoothing_window = st.number_input("Окно сглаживания", min_value=1, max_value=51, value=5, step=2)
        zoom_to_trajectory = st.checkbox("Приближать к траектории", value=False)
        zoom_margin_factor = st.slider(
            "Запас вокруг маршрута при приближении",
            min_value=0.2,
            max_value=3.0,
            value=1.0,
            step=0.1,
            disabled=not zoom_to_trajectory,
        )

    if dem_path is None:
        st.warning("Выберите DEM-карту `.tif`.")
        return

    if nmea_path is None:
        st.warning("Загрузите NMEA-файл или сгенерируйте тестовые данные.")
        return

    if st.button("Запустить локализацию", type="primary"):
        with st.spinner("Считаю траектории и ошибки..."):
            result = localize_position_from_nmea(
                dem_path=dem_path,
                nmea_path=nmea_path,
                baro_altitude_m=baro_altitude_m,
                sample_rate_hz=sample_rate_hz,
                min_speed_mps=min_speed_mps,
                max_speed_mps=max_speed_mps,
                coarse_speed_step_mps=coarse_speed_step_mps,
                fine_speed_step_mps=fine_speed_step_mps,
                coarse_azimuth_step_deg=coarse_azimuth_step_deg,
                fine_azimuth_step_deg=fine_azimuth_step_deg,
                coarse_start_step_m=coarse_start_step_m,
                refine_radius_m=refine_radius_m,
                refine_start_step_m=refine_start_step_m,
                coarse_top_k=int(coarse_top_k),
                flat_variance_threshold_m2=flat_variance_threshold_m2,
                smoothing_window=int(smoothing_window),
                smoothing_method=smoothing_method,
                use_weighted_scoring=bool(use_weighted_scoring),
                coarse_profile_points=int(coarse_profile_points),
                max_profile_points=int(max_profile_points),
                search_radius_m=None if search_radius_m <= 0 else float(search_radius_m),
                auto_retry_unweighted=False,
            )

        if result.is_flat_terrain:
            st.warning("Профиль рельефа почти плоский: поиск по карте отключен, использованы старые координаты.")
        elif not use_weighted_scoring and _has_dominant_flat_sections(result.measured_profile_m):
            st.info("В профиле много плоских участков. Если результат выглядит неуверенно, попробуйте включить `Weighted scoring для плоских участков`.")

        col_speed, col_course, col_error, col_corr, col_confidence, col_accuracy = st.columns(6)
        col_speed.metric("Скорость", f"{result.speed_mps:.1f} м/с")
        col_course.metric("Курс", f"{result.azimuth_deg:.0f}°")
        error_label = "Weighted RMSE" if result.scoring_mode == "weighted" else "RMSE"
        col_error.metric(error_label, f"{result.best_error:.2f} м")
        col_corr.metric("Correlation", f"{result.best_correlation:.3f}")
        col_confidence.metric("Confidence", f"{result.confidence * 100:.0f}%")
        accuracy_text = "неопределена" if not np.isfinite(result.estimated_accuracy_m) else f"±{result.estimated_accuracy_m:.0f} м"
        col_accuracy.metric("Оценка точности", accuracy_text)

        col_start, col_current, col_velocity = st.columns(3)
        col_start.metric("Старт X/Y, м", f"{result.start_x_m:.0f}, {result.start_y_m:.0f}")
        col_current.metric("Текущие X/Y, м", f"{result.current_x_m:.0f}, {result.current_y_m:.0f}")
        col_velocity.metric("Вектор Vx/Vy", f"{result.velocity_x_mps:.2f}, {result.velocity_y_mps:.2f} м/с")

        plot_col, map_col = st.columns(2)
        with plot_col:
            st.subheader("Heatmap корреляции")
            _show_figure(_plot_correlation_heatmap(result.correlations, result.speeds_mps, result.azimuths_deg))

        with map_col:
            st.subheader("Траектория на DEM")
            st.pyplot(
                _plot_trajectory_on_dem(
                    dem_path,
                    result.smoothed_trajectory_x_m,
                    result.smoothed_trajectory_y_m,
                    zoom_to_trajectory=zoom_to_trajectory,
                    zoom_margin_factor=float(zoom_margin_factor),
                )
            )

        profile_col_1, profile_col_2 = st.columns(2)
        with profile_col_1:
            st.subheader("Профили высот")
            st.pyplot(_plot_profiles(result.measured_profile_m, result.predicted_profile_m))

        with profile_col_2:
            st.subheader("Параметры результата")
            st.write(
                {
                    "speed_mps": float(result.speed_mps),
                    "azimuth_deg": float(result.azimuth_deg),
                    "velocity_x_mps": float(result.velocity_x_mps),
                    "velocity_y_mps": float(result.velocity_y_mps),
                    "start_x_m": float(result.start_x_m),
                    "start_y_m": float(result.start_y_m),
                    "current_x_m": float(result.current_x_m),
                    "current_y_m": float(result.current_y_m),
                    "best_error": float(result.best_error),
                    "requested_weighted_scoring": bool(use_weighted_scoring),
                    "scoring_mode": result.scoring_mode,
                    "best_correlation": float(result.best_correlation),
                    "confidence": float(result.confidence),
                    "estimated_accuracy_m": float(result.estimated_accuracy_m),
                    "quality_label": result.quality_label,
                    "terrain_variance_m2": float(result.terrain_variance_m2),
                    "is_flat_terrain": bool(result.is_flat_terrain),
                    "speed_index": int(result.best_speed_index),
                    "azimuth_index": int(result.best_azimuth_index),
                }
            )


def _select_dem_path() -> str | None:
    uploaded_dem = st.file_uploader("Загрузить DEM `.tif`", type=["tif", "tiff"])

    if uploaded_dem is not None:
        return _save_uploaded_file(uploaded_dem, suffix=".tif")

    if DEFAULT_DEM_PATH.exists():
        st.success(f"Используется карта: {DEFAULT_DEM_PATH}")
        return str(DEFAULT_DEM_PATH)

    return None


def _select_nmea_path(dem_path: str | None) -> str | None:
    uploaded_nmea = st.file_uploader("Загрузить NMEA `.txt/.nmea`", type=["txt", "nmea"])
    if uploaded_nmea is not None:
        return _save_uploaded_file(uploaded_nmea, suffix=".nmea")

    st.divider()
    st.subheader("Тестовый полет")

    speed = st.number_input("Тестовая скорость, м/с", min_value=1.0, value=20.0, step=1.0)
    azimuth = st.number_input("Тестовый курс, град", min_value=0.0, max_value=359.0, value=225.0, step=1.0)
    duration = st.number_input("Длительность, с", min_value=5.0, value=600.0, step=60.0)
    rate = st.number_input("Частота теста, Гц", min_value=0.1, max_value=10.0, value=1.0, step=0.5)
    noise = st.number_input("Шум, м", min_value=0.0, value=2.0, step=0.5)
    route_seed = st.number_input("Seed маршрута", min_value=0, value=7, step=1)
    seed = st.number_input("Seed шума", min_value=0, value=42, step=1)

    if st.button("Сгенерировать NMEA"):
        if dem_path is None:
            st.error("Сначала выберите DEM-карту.")
        else:
            try:
                generated = generate_test_flight(
                    dem_path=dem_path,
                    output_path=DEFAULT_NMEA_PATH,
                    speed_mps=speed,
                    azimuth_deg=azimuth,
                    duration_s=duration,
                    sample_rate_hz=rate,
                    noise_std_m=noise,
                    seed=int(seed),
                    route_seed=int(route_seed),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["generated_nmea_path"] = str(DEFAULT_NMEA_PATH)
                st.success(f"Создано {generated.timestamps_s.size} NMEA-сообщений.")

    generated_path = st.session_state.get("generated_nmea_path")
    if generated_path:
        return generated_path

    if DEFAULT_NMEA_PATH.exists():
        st.info(f"Используется тестовый файл: {DEFAULT_NMEA_PATH}")
        return str(DEFAULT_NMEA_PATH)

    return None


def _save_uploaded_file(uploaded_file, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _has_dominant_flat_sections(profile_m: np.ndarray) -> bool:
    if profile_m.size < 20:
        return False

    fill = float(np.nanmedian(profile_m)) if np.isfinite(profile_m).any() else 0.0
    clean = np.nan_to_num(profile_m.astype(np.float64, copy=False), nan=fill)
    gradient = np.abs(np.gradient(clean))
    curvature = np.abs(np.gradient(np.gradient(clean)))
    features = gradient + 0.5 * curvature

    high = float(np.percentile(features, 90.0))
    if high <= 1e-9:
        return True

    flat_threshold = max(1.0, high * 0.15)
    flat_fraction = float(np.mean(features <= flat_threshold))
    return flat_fraction >= 0.7


def _show_figure(fig) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    st.image(buffer, use_container_width=True)


def _plot_correlation_heatmap(correlations: np.ndarray, speeds: np.ndarray, azimuths: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 5))
    if not np.isfinite(correlations).any():
        ax.text(0.5, 0.5, "Поиск отключен: плоский рельеф", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    image = ax.imshow(
        correlations,
        aspect="auto",
        origin="lower",
        extent=[azimuths.min(), azimuths.max(), speeds.min(), speeds.max()],
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_xlabel("Азимут, град")
    ax.set_ylabel("Скорость, м/с")
    ax.set_title("Корреляция по сетке скорость-азимут")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    return fig


def _plot_trajectory_on_dem(
    dem_path: str,
    trajectory_x_m: np.ndarray,
    trajectory_y_m: np.ndarray,
    zoom_to_trajectory: bool = False,
    zoom_margin_factor: float = 1.0,
):
    with rasterio.open(dem_path) as dataset:
        dem = dataset.read(1)
        source_crs = CRS.from_user_input(dataset.crs)
        center_lon, center_lat = dataset_center_lonlat(dataset, source_crs)
        utm_crs = utm_crs_for_lonlat(center_lon, center_lat)
        transformer = Transformer.from_crs(utm_crs, source_crs, always_xy=True)
        xs, ys = transformer.transform(trajectory_x_m, trajectory_y_m)
        rows, cols = rowcol(dataset.transform, xs, ys, op=np.float64)

    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(dem, cmap="terrain")
    ax.plot(cols, rows, color="red", linewidth=4.0)
    if cols.size:
        ax.scatter(cols[0], rows[0], color="white", edgecolor="red", linewidth=1.5, s=55, zorder=4)
        ax.scatter(cols[-1], rows[-1], color="red", s=60, zorder=4)
    if cols.size >= 2:
        ax.annotate(
            "",
            xy=(cols[-1], rows[-1]),
            xytext=(cols[max(0, cols.size - 4)], rows[max(0, rows.size - 4)]),
            arrowprops={"arrowstyle": "->", "linewidth": 3, "color": "red"},
        )
    if zoom_to_trajectory and cols.size:
        span = max(float(np.nanmax(cols) - np.nanmin(cols)), float(np.nanmax(rows) - np.nanmin(rows)), 100.0)
        margin = span * max(float(zoom_margin_factor), 0.2)
        ax.set_xlim(max(0.0, float(np.nanmin(cols)) - margin), min(dem.shape[1], float(np.nanmax(cols)) + margin))
        ax.set_ylim(min(dem.shape[0], float(np.nanmax(rows)) + margin), max(0.0, float(np.nanmin(rows)) - margin))
    ax.set_title("Найденная траектория")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _plot_profiles(measured: np.ndarray, predicted: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(measured, label="NMEA -> рельеф", linewidth=2)
    ax.plot(predicted, label="DEM по найденной траектории", linewidth=2)
    ax.set_xlabel("Номер измерения")
    ax.set_ylabel("Высота рельефа, м")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
