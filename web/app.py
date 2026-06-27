from __future__ import annotations

import base64
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
from terrain_nav.io.nmea import parse_nmea_profile
from terrain_nav.core.rust_core import compute_error_grid_rust
from terrain_nav.core.search import localize_position_from_nmea
from terrain_nav.core.flat_gap_search import localize_with_flat_gap_bridge
from terrain_nav.core.utm_raster import load_utm_raster
from terrain_nav.simulation import generate_test_flight


DEFAULT_DEM_PATH = ROOT_DIR / "data" / "map.tif"
DEFAULT_NMEA_PATH = Path("outputs/test_flight.nmea")
RADAR_GIF_PATH = ROOT_DIR / "data" / "tenor.gif"

PANEL_BG = "#101827"
AXIS_BG = "#0b1220"
TEXT_COLOR = "#e7edf7"
MUTED_COLOR = "#8ea0b8"
ACCENT_COLOR = "#27f58a"


def _inject_theme() -> None:
    radar_uri = _asset_data_uri(RADAR_GIF_PATH)
    radar_background = f"url('{radar_uri}')" if radar_uri else "none"
    st.markdown(
        f"""
        <style>
        :root {{
            --nav-bg: #07111f;
            --nav-panel: rgba(16, 24, 39, 0.86);
            --nav-panel-strong: rgba(11, 18, 32, 0.95);
            --nav-line: rgba(93, 245, 168, 0.22);
            --nav-text: #e7edf7;
            --nav-muted: #8ea0b8;
            --nav-accent: #27f58a;
            --nav-red: #ff4b4b;
            --nav-amber: #f5c542;
        }}

        .stApp {{
            color: var(--nav-text);
            background:
                linear-gradient(180deg, rgba(7, 17, 31, 0.96), rgba(8, 13, 24, 0.98)),
                repeating-linear-gradient(90deg, rgba(39,245,138,0.035) 0 1px, transparent 1px 96px),
                repeating-linear-gradient(0deg, rgba(39,245,138,0.028) 0 1px, transparent 1px 96px);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0b1220 0%, #101827 100%);
            border-right: 1px solid rgba(39,245,138,0.16);
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {{
            color: var(--nav-text);
        }}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 4rem;
            max-width: 1480px;
        }}

        .nav-hero {{
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(39,245,138,0.22);
            background:
                linear-gradient(115deg, rgba(10, 18, 32, 0.96) 0%, rgba(13, 28, 43, 0.91) 54%, rgba(7, 17, 31, 0.96) 100%);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 24px;
            box-shadow: 0 24px 70px rgba(0,0,0,0.36);
        }}

        .nav-hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background-image: {radar_background};
            background-repeat: no-repeat;
            background-position: right 26px center;
            background-size: min(34vw, 360px);
            opacity: 0.28;
            pointer-events: none;
        }}

        .nav-hero-content {{
            position: relative;
            z-index: 1;
            max-width: 760px;
        }}

        .nav-kicker {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--nav-accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .nav-kicker::before {{
            content: "";
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--nav-accent);
            box-shadow: 0 0 18px var(--nav-accent);
        }}

        .nav-title {{
            color: #f7fbff;
            font-size: clamp(2.4rem, 5vw, 5rem);
            line-height: 0.96;
            font-weight: 900;
            margin: 0;
            letter-spacing: 0;
        }}

        .nav-subtitle {{
            color: var(--nav-muted);
            font-size: 1.05rem;
            max-width: 700px;
            margin: 16px 0 20px;
        }}

        .nav-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .nav-chip {{
            border: 1px solid rgba(39,245,138,0.26);
            background: rgba(39,245,138,0.08);
            color: #dfffee;
            padding: 8px 11px;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 650;
        }}

        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, rgba(16,24,39,0.96), rgba(9,15,27,0.94));
            border: 1px solid rgba(39,245,138,0.18);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 14px 34px rgba(0,0,0,0.26);
        }}

        div[data-testid="stMetricLabel"] p {{
            color: var(--nav-muted);
            font-weight: 700;
        }}

        div[data-testid="stMetricValue"] {{
            color: #f3f8ff;
        }}

        h1, h2, h3 {{
            color: #f4f8ff;
            letter-spacing: 0;
        }}

        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdownContainer"] h3) {{
            border-color: rgba(39,245,138,0.12);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid rgba(39,245,138,0.18);
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(16,24,39,0.8);
            border: 1px solid rgba(39,245,138,0.14);
            border-bottom: none;
            border-radius: 10px 10px 0 0;
            color: var(--nav-muted);
            padding: 10px 18px;
        }}

        .stTabs [aria-selected="true"] {{
            color: #ffffff;
            border-color: rgba(39,245,138,0.38);
            background: rgba(39,245,138,0.12);
        }}

        .stButton > button {{
            border-radius: 12px;
            border: 1px solid rgba(39,245,138,0.34);
            background: linear-gradient(90deg, #18c66f, #2ef5a0);
            color: #03110b;
            font-weight: 800;
            box-shadow: 0 14px 28px rgba(39,245,138,0.18);
        }}

        .stButton > button:hover {{
            border-color: rgba(39,245,138,0.8);
            color: #03110b;
            filter: brightness(1.05);
        }}

        div[data-testid="stAlert"] {{
            border-radius: 14px;
            border: 1px solid rgba(39,245,138,0.16);
            background: rgba(16,24,39,0.86);
            color: var(--nav-text);
        }}

        [data-testid="stFileUploader"] section {{
            background: rgba(7,17,31,0.72);
            border: 1px dashed rgba(39,245,138,0.24);
            border-radius: 14px;
        }}

        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"] {{
            background-color: rgba(7,17,31,0.88);
            border-color: rgba(39,245,138,0.18);
            color: var(--nav-text);
        }}

        @media (max-width: 900px) {{
            .nav-hero {{
                padding: 22px;
            }}
            .nav-hero::after {{
                opacity: 0.16;
                background-size: 260px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _asset_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    mime = "image/gif" if path.suffix.lower() == ".gif" else "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _render_hero() -> None:
    st.markdown(
        """
        <section class="nav-hero">
            <div class="nav-hero-content">
                <div class="nav-kicker">GNSS-denied terrain navigation</div>
                <h1 class="nav-title">Полет вслепую</h1>
                <p class="nav-subtitle">
                    Резервная навигация по радиовысотомеру и DEM: координаты, скорость,
                    курс и уверенность результата без спутникового сигнала.
                </p>
                <div class="nav-chips">
                    <span class="nav-chip">Rust-core</span>
                    <span class="nav-chip">DEM matching</span>
                    <span class="nav-chip">Correlation 0-359°</span>
                    <span class="nav-chip">Kalman smoothing</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Полет вслепую", layout="wide")
    _inject_theme()
    _render_hero()

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
        coarse_start_step_m = st.number_input("Грубый шаг X/Y, м", min_value=500.0, value=1000.0, step=500.0)
        refine_radius_m = st.number_input("Радиус уточнения X/Y, м", min_value=0.0, value=10000.0, step=500.0)
        refine_start_step_m = st.number_input("Точный шаг X/Y, м", min_value=100.0, value=1000.0, step=100.0)
        coarse_top_k = st.number_input("Кандидатов для уточнения", min_value=1, max_value=300, value=200, step=10)
        coarse_profile_points = st.number_input("Точек для грубого поиска", min_value=50, max_value=1000, value=250, step=10)
        max_profile_points = st.number_input("Точек профиля для поиска", min_value=50, max_value=5000, value=600, step=50)
        search_radius_m = st.number_input("Радиус поиска от центра, м (0 = вся карта)", min_value=0.0, value=0.0, step=5000.0)
        flat_variance_threshold_m2 = st.number_input("Порог плоского рельефа, м²", min_value=0.0, value=1.0, step=0.5)
        selected_algorithms = st.multiselect(
            "Алгоритмы для запуска",
            ["RMSE", "Weighted scoring", "Bridge mode"],
            default=["RMSE"],
        )
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
        if not selected_algorithms:
            st.warning("Выберите хотя бы один алгоритм.")
            return

        with st.spinner("Считаю траектории и ошибки..."):
            common_kwargs = {
                "min_speed_mps": min_speed_mps,
                "max_speed_mps": max_speed_mps,
                "coarse_speed_step_mps": coarse_speed_step_mps,
                "fine_speed_step_mps": fine_speed_step_mps,
                "coarse_azimuth_step_deg": coarse_azimuth_step_deg,
                "fine_azimuth_step_deg": fine_azimuth_step_deg,
                "coarse_start_step_m": coarse_start_step_m,
                "refine_radius_m": refine_radius_m,
                "refine_start_step_m": refine_start_step_m,
                "coarse_top_k": int(coarse_top_k),
                "flat_variance_threshold_m2": flat_variance_threshold_m2,
                "smoothing_window": int(smoothing_window),
                "smoothing_method": smoothing_method,
                "coarse_profile_points": int(coarse_profile_points),
                "max_profile_points": int(max_profile_points),
                "search_radius_m": None if search_radius_m <= 0 else float(search_radius_m),
            }
            results = []
            for algorithm in selected_algorithms:
                if algorithm == "Bridge mode":
                    try:
                        bridge_result = localize_with_flat_gap_bridge(
                            dem_path=dem_path,
                            nmea_path=nmea_path,
                            baro_altitude_m=baro_altitude_m,
                            sample_rate_hz=sample_rate_hz,
                            flat_window_points=max(5, int(sample_rate_hz * 31)),
                            gap_variance_threshold_m2=max(float(flat_variance_threshold_m2), 25.0),
                            gap_gradient_threshold_m=8.0,
                            min_flat_duration_s=60.0,
                            min_gap_start_fraction=0.45,
                            use_weighted_scoring=False,
                            **common_kwargs,
                        )
                    except ValueError as exc:
                        results.append((algorithm, exc))
                    else:
                        results.append((algorithm, bridge_result))
                    continue

                result = localize_position_from_nmea(
                    dem_path=dem_path,
                    nmea_path=nmea_path,
                    baro_altitude_m=baro_altitude_m,
                    sample_rate_hz=sample_rate_hz,
                    use_weighted_scoring=algorithm == "Weighted scoring",
                    auto_retry_unweighted=False,
                    **common_kwargs,
                )
                results.append((algorithm, result))

        if len(results) == 1:
            _render_algorithm_result(
                results[0][0],
                results[0][1],
                dem_path,
                nmea_path,
                float(baro_altitude_m),
                float(sample_rate_hz),
                int(max_profile_points),
                float(min_speed_mps),
                float(max_speed_mps),
                float(fine_speed_step_mps),
                zoom_to_trajectory,
                float(zoom_margin_factor),
            )
            return

        tabs = st.tabs([name for name, _ in results])
        for tab, (algorithm, result) in zip(tabs, results):
            with tab:
                _render_algorithm_result(
                    algorithm,
                    result,
                    dem_path,
                    nmea_path,
                    float(baro_altitude_m),
                    float(sample_rate_hz),
                    int(max_profile_points),
                    float(min_speed_mps),
                    float(max_speed_mps),
                    float(fine_speed_step_mps),
                    zoom_to_trajectory,
                    float(zoom_margin_factor),
                )


def _select_dem_path() -> str | None:
    uploaded_dem = st.file_uploader("Загрузить DEM `.tif`", type=["tif", "tiff"])

    if uploaded_dem is not None:
        return _save_uploaded_file(uploaded_dem, suffix=".tif")

    if DEFAULT_DEM_PATH.exists():
        st.success(f"Используется карта: {DEFAULT_DEM_PATH}")
        return str(DEFAULT_DEM_PATH)

    return None


def _render_algorithm_result(
    algorithm: str,
    result,
    dem_path: str,
    nmea_path: str,
    baro_altitude_m: float,
    sample_rate_hz: float,
    max_profile_points: int,
    full_heatmap_min_speed_mps: float,
    full_heatmap_max_speed_mps: float,
    full_heatmap_speed_step_mps: float,
    zoom_to_trajectory: bool,
    zoom_margin_factor: float,
) -> None:
    if isinstance(result, Exception):
        if algorithm == "Bridge mode":
            st.info(f"{algorithm}: {result}")
        else:
            st.error(f"{algorithm}: {result}")
        return

    if algorithm == "Bridge mode":
        _render_bridge_result(
            dem_path=dem_path,
            nmea_path=nmea_path,
            baro_altitude_m=baro_altitude_m,
            sample_rate_hz=sample_rate_hz,
            max_profile_points=max_profile_points,
            full_heatmap_min_speed_mps=full_heatmap_min_speed_mps,
            full_heatmap_max_speed_mps=full_heatmap_max_speed_mps,
            full_heatmap_speed_step_mps=full_heatmap_speed_step_mps,
            bridge_result=result,
            zoom_to_trajectory=zoom_to_trajectory,
            zoom_margin_factor=zoom_margin_factor,
        )
        return

    _render_localization_result(
        algorithm=algorithm,
        result=result,
        dem_path=dem_path,
        nmea_path=nmea_path,
        baro_altitude_m=baro_altitude_m,
        sample_rate_hz=sample_rate_hz,
        max_profile_points=max_profile_points,
        full_heatmap_min_speed_mps=full_heatmap_min_speed_mps,
        full_heatmap_max_speed_mps=full_heatmap_max_speed_mps,
        full_heatmap_speed_step_mps=full_heatmap_speed_step_mps,
        zoom_to_trajectory=zoom_to_trajectory,
        zoom_margin_factor=zoom_margin_factor,
    )


def _render_localization_result(
    algorithm: str,
    result,
    dem_path: str,
    nmea_path: str,
    baro_altitude_m: float,
    sample_rate_hz: float,
    max_profile_points: int,
    full_heatmap_min_speed_mps: float,
    full_heatmap_max_speed_mps: float,
    full_heatmap_speed_step_mps: float,
    zoom_to_trajectory: bool,
    zoom_margin_factor: float,
) -> None:
    if result.is_flat_terrain:
        st.warning("Профиль рельефа почти плоский: поиск по карте отключен, использованы старые координаты.")
    elif algorithm == "RMSE" and _has_dominant_flat_sections(result.measured_profile_m):
        st.info("В профиле много плоских участков. Если результат выглядит неуверенно, попробуйте `Weighted scoring` или `Bridge mode`.")

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
        st.subheader("Heatmap корреляции 0-359°")
        _render_full_direction_heatmap(
            dem_path=dem_path,
            nmea_path=nmea_path,
            baro_altitude_m=baro_altitude_m,
            sample_rate_hz=sample_rate_hz,
            max_profile_points=max_profile_points,
            result=result,
            min_speed_mps=full_heatmap_min_speed_mps,
            max_speed_mps=full_heatmap_max_speed_mps,
            speed_step_mps=full_heatmap_speed_step_mps,
            use_weighted_scoring=algorithm == "Weighted scoring",
        )

    with map_col:
        st.subheader("Траектория на DEM")
        st.pyplot(
            _plot_trajectory_on_dem(
                dem_path,
                result.smoothed_trajectory_x_m,
                result.smoothed_trajectory_y_m,
                zoom_to_trajectory=zoom_to_trajectory,
                zoom_margin_factor=zoom_margin_factor,
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
                "algorithm": algorithm,
                "speed_mps": float(result.speed_mps),
                "azimuth_deg": float(result.azimuth_deg),
                "velocity_x_mps": float(result.velocity_x_mps),
                "velocity_y_mps": float(result.velocity_y_mps),
                "start_x_m": float(result.start_x_m),
                "start_y_m": float(result.start_y_m),
                "current_x_m": float(result.current_x_m),
                "current_y_m": float(result.current_y_m),
                "best_error": float(result.best_error),
                "requested_weighted_scoring": algorithm == "Weighted scoring",
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


def _render_bridge_result(
    dem_path: str,
    nmea_path: str,
    baro_altitude_m: float,
    sample_rate_hz: float,
    max_profile_points: int,
    full_heatmap_min_speed_mps: float,
    full_heatmap_max_speed_mps: float,
    full_heatmap_speed_step_mps: float,
    bridge_result,
    zoom_to_trajectory: bool,
    zoom_margin_factor: float,
) -> None:
    if bridge_result.has_flat_gap:
        st.info(
            "Bridge mode: найден слабый/плоский участок профиля. Алгоритм берет уверенный фрагмент до него, "
            "продолжает прямую траекторию и проверяет совпадение после участка."
        )
    else:
        st.info(
            "Bridge mode: весь профиль информативен. Алгоритм строит наиболее вероятную единую прямую "
            "траекторию по DEM и использует весь рельеф для подтверждения направления."
        )
    if bridge_result.line.confidence < 0.35:
        st.warning(
            "Опорный фрагмент дает низкую уверенность. Направление найдено как гипотеза, "
            "но его нужно проверять по RMSE, корреляции и карте."
        )
    elif bridge_result.confidence < 0.35:
        st.warning("Проверочный участок слабый или продолжение слишком длинное: confidence заметно снизился.")
    if not bridge_result.is_reliable:
        st.warning(
            "Bridge mode выдал наиболее вероятную гипотезу, но надежность низкая: "
            f"{bridge_result.rejection_reason}."
        )

    col_speed, col_course, col_history, col_gap, col_check, col_accuracy = st.columns(6)
    col_speed.metric("Скорость входа", f"{bridge_result.speed_mps:.1f} м/с")
    col_course.metric("Курс входа", f"{bridge_result.azimuth_deg:.0f}°")
    col_history.metric("Опорный фрагмент", f"{bridge_result.before_point_count} точек")
    col_gap.metric("Слабый участок", f"{bridge_result.gap.duration_s:.0f} с" if bridge_result.has_flat_gap else "нет")
    col_check.metric("Проверка" if bridge_result.has_flat_gap else "RMSE профиля", f"{bridge_result.after_validation_rmse_m:.1f} м")
    col_accuracy.metric("Оценка точности", f"±{bridge_result.estimated_accuracy_m:.0f} м")

    col_start, col_current, col_velocity, col_confidence = st.columns(4)
    col_start.metric("Старт X/Y, м", f"{bridge_result.start_x_m:.0f}, {bridge_result.start_y_m:.0f}")
    col_current.metric("Текущие X/Y, м", f"{bridge_result.current_x_m:.0f}, {bridge_result.current_y_m:.0f}")
    col_velocity.metric(
        "Вектор Vx/Vy",
        f"{bridge_result.velocity_x_mps:.2f}, {bridge_result.velocity_y_mps:.2f} м/с",
    )
    col_confidence.metric("Bridge confidence", f"{bridge_result.confidence * 100:.0f}%")

    heat_col, map_col = st.columns(2)
    with heat_col:
        st.subheader("Heatmap корреляции 0-359°")
        _render_full_direction_heatmap(
            dem_path=dem_path,
            nmea_path=nmea_path,
            baro_altitude_m=baro_altitude_m,
            sample_rate_hz=sample_rate_hz,
            max_profile_points=max_profile_points,
            result=bridge_result.line,
            min_speed_mps=full_heatmap_min_speed_mps,
            max_speed_mps=full_heatmap_max_speed_mps,
            speed_step_mps=full_heatmap_speed_step_mps,
            use_weighted_scoring=False,
            profile_override=(
                bridge_result.line.measured_profile_m,
                _timestamps_like_profile(bridge_result.line.measured_profile_m, sample_rate_hz),
            ),
        )
    with map_col:
        st.subheader("Прямая траектория на DEM")
        st.pyplot(
            _plot_bridge_trajectory_on_dem(
                dem_path,
                bridge_result,
                zoom_to_trajectory,
                zoom_margin_factor,
                accepted=bridge_result.is_reliable,
            )
        )

    profile_col, params_col = st.columns(2)
    with profile_col:
        st.subheader("Информативные сегменты")
        st.pyplot(_plot_bridge_segments(bridge_result))
    with params_col:
        st.subheader("Параметры режима продолжения")
        st.write(
            {
                "mode": "trajectory_bridge",
                "has_flat_gap": bool(bridge_result.has_flat_gap),
                "line_speed_mps": float(bridge_result.line.speed_mps),
                "line_azimuth_deg": float(bridge_result.line.azimuth_deg),
                "line_confidence": float(bridge_result.line.confidence),
                "line_rmse": float(bridge_result.line.best_error),
                "line_correlation": float(bridge_result.line.best_correlation),
                "after_validation_rmse_m": float(bridge_result.after_validation_rmse_m),
                "after_validation_correlation": float(bridge_result.after_validation_correlation),
                "before_points_used": int(bridge_result.before_point_count),
                "after_points_used": int(bridge_result.after_point_count),
                "gap_start_index": int(bridge_result.gap.start_index),
                "gap_end_index": int(bridge_result.gap.end_index),
                "gap_duration_s": float(bridge_result.gap.duration_s),
                "gap_variance_m2": float(bridge_result.gap.variance_m2),
                "bridge_correction_x_m": float(bridge_result.bridge.correction_x_m),
                "bridge_correction_y_m": float(bridge_result.bridge.correction_y_m),
                "after_anchor_cross_track_error_m": float(bridge_result.after_anchor_cross_track_error_m),
                "after_anchor_along_track_error_m": float(bridge_result.after_anchor_along_track_error_m),
                "bridge_confidence": float(bridge_result.confidence),
                "estimated_accuracy_m": float(bridge_result.estimated_accuracy_m),
                "quality_label": bridge_result.quality_label,
                "is_reliable": bool(bridge_result.is_reliable),
                "rejection_reason": bridge_result.rejection_reason,
            }
        )


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


def _render_full_direction_heatmap(
    dem_path: str,
    nmea_path: str,
    baro_altitude_m: float,
    sample_rate_hz: float,
    max_profile_points: int,
    result,
    min_speed_mps: float,
    max_speed_mps: float,
    speed_step_mps: float,
    use_weighted_scoring: bool,
    profile_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    try:
        if profile_override is None:
            measured_profile, timestamps = _load_limited_profile_for_heatmap(
                nmea_path=nmea_path,
                baro_altitude_m=baro_altitude_m,
                sample_rate_hz=sample_rate_hz,
                max_profile_points=max_profile_points,
            )
        else:
            measured_profile, timestamps = profile_override
        speeds, azimuths, correlations = _compute_full_direction_heatmap_data(
            dem_path=str(dem_path),
            measured_profile_m=tuple(float(value) for value in measured_profile),
            timestamps_s=tuple(float(value) for value in timestamps),
            start_x_m=float(result.start_x_m),
            start_y_m=float(result.start_y_m),
            min_speed_mps=float(min_speed_mps),
            max_speed_mps=float(max_speed_mps),
            speed_step_mps=float(speed_step_mps),
            use_weighted_scoring=bool(use_weighted_scoring),
        )
    except Exception as exc:
        st.warning(f"Полную heatmap 0-359° построить не удалось, показана локальная сетка: {exc}")
        _show_figure(
            _plot_correlation_heatmap(
                result.correlations,
                result.speeds_mps,
                result.azimuths_deg,
                title="Локальная корреляция финального уточнения",
            )
        )
        return

    _show_figure(
        _plot_correlation_heatmap(
            correlations,
            speeds,
            azimuths,
            title="Полная корреляция по направлениям 0-359°",
        )
    )


def _load_limited_profile_for_heatmap(
    nmea_path: str,
    baro_altitude_m: float,
    sample_rate_hz: float,
    max_profile_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    profile = parse_nmea_profile(nmea_path, baro_altitude_m, sample_rate_hz)
    if max_profile_points > 0 and profile.timestamps_s.size > max_profile_points:
        indexes = np.unique(np.linspace(0, profile.timestamps_s.size - 1, int(max_profile_points), dtype=int))
        return profile.terrain_profile_m[indexes], profile.timestamps_s[indexes]
    return profile.terrain_profile_m, profile.timestamps_s


def _timestamps_like_profile(profile_m: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    step_s = 1.0 / max(float(sample_rate_hz), 1e-9)
    return np.arange(np.asarray(profile_m).size, dtype=np.float64) * step_s


@st.cache_data(show_spinner=False, max_entries=8)
def _compute_full_direction_heatmap_data(
    dem_path: str,
    measured_profile_m: tuple[float, ...],
    timestamps_s: tuple[float, ...],
    start_x_m: float,
    start_y_m: float,
    min_speed_mps: float,
    max_speed_mps: float,
    speed_step_mps: float,
    use_weighted_scoring: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed_step = max(float(speed_step_mps), 0.1)
    speeds = np.arange(float(min_speed_mps), float(max_speed_mps) + speed_step * 0.5, speed_step, dtype=np.float64)
    azimuths = np.arange(0.0, 360.0, 1.0, dtype=np.float64)
    measured = np.asarray(measured_profile_m, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)

    with rasterio.open(dem_path) as dataset:
        utm_raster = load_utm_raster(dataset)

    _, correlations, _ = compute_error_grid_rust(
        dem=utm_raster.heights,
        transform=utm_raster.transform,
        measured_profile_m=measured,
        timestamps_s=timestamps,
        start_x_m=float(start_x_m),
        start_y_m=float(start_y_m),
        speeds_mps=speeds,
        azimuths_deg=azimuths,
        use_weighted_scoring=bool(use_weighted_scoring),
    )
    return speeds, azimuths, correlations


def _plot_correlation_heatmap(
    correlations: np.ndarray,
    speeds: np.ndarray,
    azimuths: np.ndarray,
    title: str = "Корреляция по сетке скорость-азимут",
):
    fig, ax = plt.subplots(figsize=(8, 5))
    _style_figure(fig, ax)
    if not np.isfinite(correlations).any():
        ax.text(0.5, 0.5, "Поиск отключен: плоский рельеф", ha="center", va="center", color=TEXT_COLOR)
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
    ax.set_xlabel("Азимут, град", color=TEXT_COLOR)
    ax.set_ylabel("Скорость, м/с", color=TEXT_COLOR)
    ax.set_title(title, color=TEXT_COLOR)
    colorbar = fig.colorbar(image, ax=ax, label="Correlation")
    colorbar.ax.yaxis.label.set_color(TEXT_COLOR)
    colorbar.ax.tick_params(colors=MUTED_COLOR)
    colorbar.outline.set_edgecolor("#26364d")
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
    _style_figure(fig, ax)
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
    ax.set_title("Найденная траектория", color=TEXT_COLOR)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _plot_bridge_trajectory_on_dem(
    dem_path: str,
    bridge_result,
    zoom_to_trajectory: bool = False,
    zoom_margin_factor: float = 1.0,
    accepted: bool = True,
):
    trajectory_x_m = bridge_result.trajectory_x_m
    trajectory_y_m = bridge_result.trajectory_y_m
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
    before_end = min(int(bridge_result.before_end_index), cols.size - 1)
    gap_start = min(int(bridge_result.gap.start_index), cols.size - 1)
    gap_end = min(int(bridge_result.gap.end_index), cols.size - 1)
    after_start = min(int(bridge_result.after_start_index), cols.size - 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    _style_figure(fig, ax)
    ax.imshow(dem, cmap="terrain")
    if not accepted:
        ax.plot(cols, rows, color="#666666", linewidth=3.0, linestyle="--", label="Гипотеза низкой уверенности")
    elif not bridge_result.has_flat_gap:
        ax.plot(cols, rows, color="red", linewidth=4.0, label="Единая прямая траектория")
    elif before_end >= 1:
        ax.plot(cols[: before_end + 1], rows[: before_end + 1], color="#2ca02c", linewidth=4.0, label="Опорный фрагмент")
    if accepted and gap_end > gap_start:
        ax.plot(cols[gap_start : gap_end + 1], rows[gap_start : gap_end + 1], color="#1f77b4", linewidth=4.0, label="Продолжение")
    if accepted and after_start < cols.size - 1:
        ax.plot(cols[after_start:], rows[after_start:], color="red", linewidth=4.0, label="Проверочный участок")

    if accepted and cols.size:
        ax.scatter(cols[0], rows[0], color="white", edgecolor="#2ca02c", linewidth=1.5, s=55, zorder=4)
        ax.scatter(cols[-1], rows[-1], color="red", s=60, zorder=4)
    if accepted and cols.size >= 2:
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
    ax.set_title("Единая прямая траектория" if accepted else "Гипотеза низкой уверенности", color=TEXT_COLOR)
    ax.set_axis_off()
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _plot_profiles(measured: np.ndarray, predicted: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 5))
    _style_figure(fig, ax)
    ax.plot(measured, label="NMEA -> рельеф", linewidth=2.2, color="#48a7ff")
    ax.plot(predicted, label="DEM по найденной траектории", linewidth=2.2, color=ACCENT_COLOR)
    ax.set_xlabel("Номер измерения", color=TEXT_COLOR)
    ax.set_ylabel("Высота рельефа, м", color=TEXT_COLOR)
    _style_legend(ax)
    ax.grid(True, alpha=0.22, color="#5b6f8a")
    fig.tight_layout()
    return fig


def _plot_bridge_segments(bridge_result):
    fig, ax = plt.subplots(figsize=(8, 5))
    _style_figure(fig, ax)
    if not bridge_result.has_flat_gap:
        ax.plot(bridge_result.line.measured_profile_m, label="NMEA -> рельеф", linewidth=2.2, color="#48a7ff")
        ax.plot(bridge_result.line.predicted_profile_m, label="DEM по прямой траектории", linewidth=2.2, color=ACCENT_COLOR)
        ax.set_xlabel("Номер измерения", color=TEXT_COLOR)
        ax.set_ylabel("Высота рельефа, м", color=TEXT_COLOR)
        _style_legend(ax)
        ax.grid(True, alpha=0.22, color="#5b6f8a")
        fig.tight_layout()
        return fig

    before_size = bridge_result.before_point_count
    after_size = bridge_result.after_point_count
    measured = bridge_result.line.measured_profile_m
    predicted = bridge_result.line.predicted_profile_m
    before_measured = measured[:before_size]
    before_predicted = predicted[:before_size]
    after_measured = measured[before_size:]
    after_predicted = predicted[before_size:]
    ax.plot(
        np.arange(before_size),
        before_measured,
        label="Опорный фрагмент: NMEA -> рельеф",
        linewidth=2,
        color="#48a7ff",
    )
    ax.plot(
        np.arange(before_size),
        before_predicted,
        label="Опорный фрагмент: DEM",
        linewidth=2,
        color=ACCENT_COLOR,
    )
    offset = before_size + bridge_result.gap.end_index - bridge_result.gap.start_index + 1
    ax.plot(
        offset + np.arange(after_measured.size),
        after_measured,
        label="Проверочный участок: NMEA -> рельеф",
        linewidth=2,
        color="#ff6b6b",
    )
    ax.plot(
        offset + np.arange(after_predicted.size),
        after_predicted,
        label="Проверочный участок: DEM",
        linewidth=2,
        color="#f5c542",
    )
    ax.axvspan(before_size, offset, color="orange", alpha=0.2, label="Продолжение")
    ax.set_xlabel("Номер измерения", color=TEXT_COLOR)
    ax.set_ylabel("Высота рельефа, м", color=TEXT_COLOR)
    _style_legend(ax)
    ax.grid(True, alpha=0.22, color="#5b6f8a")
    fig.tight_layout()
    return fig


def _style_figure(fig, ax) -> None:
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(AXIS_BG)
    ax.tick_params(colors=MUTED_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#26364d")


def _style_legend(ax) -> None:
    legend = ax.legend()
    if legend is None:
        return
    legend.get_frame().set_facecolor("#0b1220")
    legend.get_frame().set_edgecolor("#26364d")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)


if __name__ == "__main__":
    main()
