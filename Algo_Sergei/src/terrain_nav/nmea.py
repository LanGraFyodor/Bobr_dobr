from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from terrain_nav.models import NmeaProfile


def parse_nmea_profile(
    nmea_path: str | Path,
    baro_altitude_m: float = 1500.0,
    sample_rate_hz: float | None = None,
) -> NmeaProfile:
    radio_altitudes: list[float] = []
    timestamps: list[float | None] = []

    with Path(nmea_path).open("r", encoding="utf-8") as file:
        for line in file:
            parsed = parse_gga_altitude(line.strip())
            if parsed is None:
                continue

            timestamp_s, altitude_m = parsed
            timestamps.append(timestamp_s)
            radio_altitudes.append(altitude_m)

    if not radio_altitudes:
        raise ValueError(f"No GPGGA altitude values found in {nmea_path}")

    radio = np.asarray(radio_altitudes, dtype=np.float64)
    terrain = baro_altitude_m - radio
    time = normalize_timestamps(timestamps, len(radio), sample_rate_hz)

    return NmeaProfile(
        radio_altitudes_m=radio,
        terrain_profile_m=terrain,
        timestamps_s=time,
    )


def parse_gga_altitude(line: str) -> tuple[float | None, float] | None:
    if not line or not line.startswith("$"):
        return None

    payload = line[1:].split("*", 1)[0]
    fields = payload.split(",")

    if not fields or not fields[0].endswith("GGA"):
        return None
    if len(fields) <= 10:
        return None

    altitude_text = fields[9]
    altitude_unit = fields[10]
    if not altitude_text or altitude_unit != "M":
        return None

    return parse_nmea_time(fields[1]), float(altitude_text)


def parse_nmea_time(value: str) -> float | None:
    if not value:
        return None

    try:
        hours = int(value[0:2])
        minutes = int(value[2:4])
        seconds = float(value[4:])
    except ValueError:
        return None

    return hours * 3600.0 + minutes * 60.0 + seconds


def normalize_timestamps(
    timestamps: Iterable[float | None],
    count: int,
    sample_rate_hz: float | None,
) -> np.ndarray:
    values = list(timestamps)
    if all(timestamp is not None for timestamp in values):
        time = np.asarray(values, dtype=np.float64)
        time -= time[0]
        return time

    if sample_rate_hz is None or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive when NMEA timestamps are missing")

    return np.arange(count, dtype=np.float64) / sample_rate_hz


_parse_gga_altitude = parse_gga_altitude
_parse_nmea_time = parse_nmea_time
_normalize_timestamps = normalize_timestamps
