"""
Optional game-day weather at the home ballpark (Open-Meteo, no API key).

Used as coarse context (wind / rain / temperature). Failures return NaNs so the imputer ignores them.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
import requests

from utils.mlb_ballpark_coords import coords_for_home_fg


@lru_cache(maxsize=384)
def _daily_weather(lat: float, lon: float, d_iso: str) -> tuple[float, float, float]:
    """
    Return (wind_mph_max, precip_inches_sum, temp_f_max) for the calendar day at lat/lon.
    """
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
        "start_date": d_iso,
        "end_date": d_iso,
    }
    try:
        d = date.fromisoformat(d_iso)
        if d < date.today():
            url = "https://archive-api.open-meteo.com/v1/archive"
        else:
            url = "https://api.open-meteo.com/v1/forecast"
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        js = r.json()
        day = js.get("daily") or {}
        tmax = (day.get("temperature_2m_max") or [float("nan")])[0]
        pr = (day.get("precipitation_sum") or [float("nan")])[0]
        wk = (day.get("wind_speed_10m_max") or [float("nan")])[0]  # km/h
        wind_mph = float(wk) * 0.621371 if np.isfinite(wk) else float("nan")
        precip_in = float(pr) / 25.4 if np.isfinite(pr) else float("nan")
        temp_f = float(tmax) * 9.0 / 5.0 + 32.0 if np.isfinite(tmax) else float("nan")
        return wind_mph, precip_in, temp_f
    except Exception:
        return float("nan"), float("nan"), float("nan")


def weather_features_for_home_park(home_fg: str | None, official_day: date | None) -> dict[str, float]:
    out = {
        "venue_wind_mph": float("nan"),
        "venue_precip_in": float("nan"),
        "venue_temp_f": float("nan"),
    }
    if official_day is None:
        return out
    ll = coords_for_home_fg(home_fg)
    if ll is None:
        return out
    lat, lon = ll
    w, p, t = _daily_weather(round(lat, 4), round(lon, 4), official_day.isoformat())
    out["venue_wind_mph"] = w
    out["venue_precip_in"] = p
    out["venue_temp_f"] = t
    return out
