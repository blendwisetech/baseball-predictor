"""
High-impact game context from MLB Stats API + optional weather:

- Rest days since each side's previous regular-season game (calendar gap − 1, min 0).
- Whether each side's previous game was on the road (travel proxy).
- 40-man injured-list counts (pitchers vs position players) from roster status codes.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import requests

from utils.game_day_weather import weather_features_for_home_park
from utils.mlb_api import MLB_STATS_BASE

IL_STATUS_CODES = frozenset({"D7", "D10", "D15", "D60"})


def _get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    r = requests.get(url, params=params or {}, timeout=35)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=256)
def _roster_summary(team_id: int, day_iso: str) -> tuple[int, int, int]:
    """
    One 40-man roster fetch: (IL_pitchers, IL_hitters, pitcher_depth).

    ``pitcher_depth`` counts pitchers on the 40-man not on the IL and not reassigned to minors
    (coarse bullpen / staff depth proxy — not recent usage).
    """
    data = _get(f"{MLB_STATS_BASE}/teams/{int(team_id)}/roster", {"rosterType": "40Man"})
    ip_il, hit_il, p_depth = 0, 0, 0
    for item in data.get("roster") or []:
        code = ((item.get("status") or {}).get("code") or "").strip().upper()
        pos_type = ((item.get("position") or {}).get("type") or "").strip()
        if code in IL_STATUS_CODES:
            if pos_type == "Pitcher":
                ip_il += 1
            else:
                hit_il += 1
        if pos_type == "Pitcher" and code not in IL_STATUS_CODES and code != "RM":
            p_depth += 1
    return ip_il, hit_il, p_depth


@lru_cache(maxsize=256)
def _team_schedule_entries(team_id: int, season: int, start_iso: str, end_iso: str) -> tuple[tuple[str, int, int], ...]:
    """
    Regular-season games for team between dates (inclusive).
    Each tuple: (official_date_iso, game_pk, was_away_int 0/1).
    """
    params = {
        "sportId": "1",
        "teamId": int(team_id),
        "startDate": start_iso,
        "endDate": end_iso,
    }
    raw = _get(f"{MLB_STATS_BASE}/schedule", params)
    rows: list[tuple[str, int, int]] = []
    for day in raw.get("dates") or []:
        d_iso = str(day.get("date") or "")
        for g in day.get("games") or []:
            if str(g.get("gameType") or "") != "R":
                continue
            st = (g.get("status") or {}).get("abstractGameState") or ""
            if st == "Cancelled":
                continue
            teams = g.get("teams") or {}
            hid = (teams.get("home") or {}).get("team", {}).get("id")
            aid = (teams.get("away") or {}).get("team", {}).get("id")
            if hid is None or aid is None:
                continue
            pk = int(g.get("gamePk") or 0)
            if pk <= 0:
                continue
            off = str(g.get("officialDate") or d_iso or "")[:10]
            if len(off) < 10:
                continue
            was_away = 1 if int(aid) == int(team_id) else 0
            rows.append((off, pk, was_away))
    rows.sort(key=lambda t: (t[0], t[1]))
    return tuple(rows)


def _rest_and_prev_away(
    team_id: int,
    season: int,
    slate_date: date,
    current_official: date,
    current_pk: int,
) -> tuple[float, float]:
    start_d = max(date(season, 3, 15), slate_date - timedelta(days=55))
    entries = _team_schedule_entries(team_id, season, start_d.isoformat(), slate_date.isoformat())
    cur_s = current_official.isoformat()
    prior = [t for t in entries if (t[0] < cur_s) or (t[0] == cur_s and t[1] < int(current_pk))]
    if not prior:
        return float("nan"), float("nan")
    prev_date_s, _prev_pk, prev_away = prior[-1]
    try:
        prev_d = date.fromisoformat(prev_date_s[:10])
    except ValueError:
        return float("nan"), float("nan")
    rest = max(0, (current_official - prev_d).days - 1)
    return float(rest), float(prev_away)


def _official_date_from_row(row: pd.Series) -> date | None:
    od = row.get("official_date")
    if od is not None and not (isinstance(od, float) and np.isnan(od)):
        s = str(od)[:10]
        try:
            return date.fromisoformat(s)
        except ValueError:
            pass
    gd = row.get("gameDate")
    if gd is None:
        return None
    try:
        return pd.to_datetime(gd, errors="coerce").date()
    except Exception:
        return None


def attach_high_impact_context(df: pd.DataFrame, season: int, slate_date: date | None) -> pd.DataFrame:
    out = df.copy()
    nan_cols = [
        "f_home_rest_days",
        "f_away_rest_days",
        "f_home_il_pitch",
        "f_home_il_hit",
        "f_away_il_pitch",
        "f_away_il_hit",
        "f_home_pitch_depth",
        "f_away_pitch_depth",
        "f_home_prev_away",
        "f_away_prev_away",
        "f_venue_wind_mph",
        "f_venue_precip_in",
        "f_venue_temp_f",
    ]
    for c in nan_cols:
        out[c] = np.nan

    if slate_date is None:
        if "gameDate" in out.columns and out["gameDate"].notna().any():
            try:
                slate_date = pd.to_datetime(out["gameDate"].dropna().iloc[0], errors="coerce").date()
            except Exception:
                slate_date = None
    if slate_date is None:
        return out

    for idx, row in out.iterrows():
        hid = row.get("home_id")
        aid = row.get("away_id")
        pk = row.get("gamePk")
        if pd.isna(hid) or pd.isna(aid) or pd.isna(pk):
            continue
        hid, aid, pk = int(hid), int(aid), int(pk)
        off_d = _official_date_from_row(row)
        if off_d is None:
            off_d = slate_date

        try:
            rh, pha = _rest_and_prev_away(hid, season, slate_date, off_d, pk)
            ra, paa = _rest_and_prev_away(aid, season, slate_date, off_d, pk)
            out.at[idx, "f_home_rest_days"] = rh
            out.at[idx, "f_away_rest_days"] = ra
            out.at[idx, "f_home_prev_away"] = pha
            out.at[idx, "f_away_prev_away"] = paa
        except Exception:
            pass

        try:
            hip, hih, hdep = _roster_summary(hid, slate_date.isoformat())
            aip, aih, adep = _roster_summary(aid, slate_date.isoformat())
            out.at[idx, "f_home_il_pitch"] = float(hip)
            out.at[idx, "f_home_il_hit"] = float(hih)
            out.at[idx, "f_away_il_pitch"] = float(aip)
            out.at[idx, "f_away_il_hit"] = float(aih)
            out.at[idx, "f_home_pitch_depth"] = float(hdep)
            out.at[idx, "f_away_pitch_depth"] = float(adep)
        except Exception:
            pass

        try:
            hf = row.get("home_fg")
            wx = weather_features_for_home_park(str(hf) if hf is not None else None, off_d)
            out.at[idx, "f_venue_wind_mph"] = wx["venue_wind_mph"]
            out.at[idx, "f_venue_precip_in"] = wx["venue_precip_in"]
            out.at[idx, "f_venue_temp_f"] = wx["venue_temp_f"]
        except Exception:
            pass

    return out
