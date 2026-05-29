"""
Ordered tabular features for game-level ML.

Baseball: offense quality (wRC+, OPS), starter run prevention (FIP, K/9) drive wins and runs.
Names are filesystem-safe (no slashes) for Parquet/CSV columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# One row per game — same order used at train time and inference time.
GAME_FEATURE_NAMES: list[str] = [
    "f_home_wrc",
    "f_away_wrc",
    "f_home_ops",
    "f_away_ops",
    "f_home_obp",
    "f_away_obp",
    "f_home_sp_fip",
    "f_away_sp_fip",
    "f_home_sp_era",
    "f_away_sp_era",
    "f_home_sp_k9",
    "f_away_sp_k9",
    "f_home_sp_bb9",
    "f_away_sp_bb9",
    "f_home_sp_hr9",
    "f_away_sp_hr9",
    # Derived / context (older CSV rows fill NaN for these columns).
    "f_run_park_idx",
    "f_wrc_diff",
    "f_ops_diff",
    "f_obp_diff",
    "f_sp_fip_diff",
    "f_sp_era_diff",
    "f_sp_k9_diff",
    # Rest / travel / injuries / weather (older rows → NaN).
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
    # Sample-confidence / stability (older CSV rows → NaN; retrain after adding).
    "f_home_team_pa_conf",
    "f_away_team_pa_conf",
    "f_home_sp_ip_conf",
    "f_away_sp_ip_conf",
]


def _f(row: pd.Series, key: str, default: float = float("nan")) -> float:
    v = row.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _conf_from_count(x: float, denom: float) -> float:
    if not np.isfinite(x) or denom <= 0:
        return float("nan")
    return float(min(1.0, max(0.0, x / denom)))


def _diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def enriched_row_to_feature_vector(row: pd.Series) -> dict[str, float]:
    """Map an enriched schedule row (MLB + team stats join) into ML feature dict."""
    h_w = _f(row, "home_team_wRC+")
    a_w = _f(row, "away_team_wRC+")
    h_ops = _f(row, "home_team_OPS")
    a_ops = _f(row, "away_team_OPS")
    h_obp = _f(row, "home_team_OBP")
    a_obp = _f(row, "away_team_OBP")
    h_fip = _f(row, "home_sp_FIP")
    a_fip = _f(row, "away_sp_FIP")
    h_era = _f(row, "home_sp_ERA")
    a_era = _f(row, "away_sp_ERA")
    h_k9 = _f(row, "home_sp_K/9")
    a_k9 = _f(row, "away_sp_K/9")
    pr = _f(row, "park_runs_factor", default=float("nan"))
    run_idx = pr / 100.0 if np.isfinite(pr) else float("nan")
    return {
        "f_home_wrc": h_w,
        "f_away_wrc": a_w,
        "f_home_ops": h_ops,
        "f_away_ops": a_ops,
        "f_home_obp": h_obp,
        "f_away_obp": a_obp,
        "f_home_sp_fip": h_fip,
        "f_away_sp_fip": a_fip,
        "f_home_sp_era": h_era,
        "f_away_sp_era": a_era,
        "f_home_sp_k9": h_k9,
        "f_away_sp_k9": a_k9,
        "f_home_sp_bb9": _f(row, "home_sp_BB/9"),
        "f_away_sp_bb9": _f(row, "away_sp_BB/9"),
        "f_home_sp_hr9": _f(row, "home_sp_HR/9"),
        "f_away_sp_hr9": _f(row, "away_sp_HR/9"),
        "f_run_park_idx": run_idx,
        "f_wrc_diff": _diff(h_w, a_w),
        "f_ops_diff": _diff(h_ops, a_ops),
        "f_obp_diff": _diff(h_obp, a_obp),
        "f_sp_fip_diff": _diff(a_fip, h_fip),
        "f_sp_era_diff": _diff(a_era, h_era),
        "f_sp_k9_diff": _diff(h_k9, a_k9),
        "f_home_rest_days": _f(row, "f_home_rest_days"),
        "f_away_rest_days": _f(row, "f_away_rest_days"),
        "f_home_il_pitch": _f(row, "f_home_il_pitch"),
        "f_home_il_hit": _f(row, "f_home_il_hit"),
        "f_away_il_pitch": _f(row, "f_away_il_pitch"),
        "f_away_il_hit": _f(row, "f_away_il_hit"),
        "f_home_pitch_depth": _f(row, "f_home_pitch_depth"),
        "f_away_pitch_depth": _f(row, "f_away_pitch_depth"),
        "f_home_prev_away": _f(row, "f_home_prev_away"),
        "f_away_prev_away": _f(row, "f_away_prev_away"),
        "f_venue_wind_mph": _f(row, "f_venue_wind_mph"),
        "f_venue_precip_in": _f(row, "f_venue_precip_in"),
        "f_venue_temp_f": _f(row, "f_venue_temp_f"),
        "f_home_team_pa_conf": _conf_from_count(_f(row, "home_team_PA"), 200.0),
        "f_away_team_pa_conf": _conf_from_count(_f(row, "away_team_PA"), 200.0),
        "f_home_sp_ip_conf": _conf_from_count(_f(row, "home_sp_IP"), 18.0),
        "f_away_sp_ip_conf": _conf_from_count(_f(row, "away_sp_IP"), 18.0),
    }


def features_dict_to_series(d: dict[str, Any]) -> pd.Series:
    return pd.Series({k: d.get(k, np.nan) for k in GAME_FEATURE_NAMES}, dtype=float)


def dataframe_X(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order feature columns; missing columns filled with NaN."""
    out = pd.DataFrame({c: df[c] if c in df.columns else np.nan for c in GAME_FEATURE_NAMES})
    return out.astype(float)
