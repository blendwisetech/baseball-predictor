"""
Pure helpers for slate display (abbreviations, score line, actual winner).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from utils.team_map import fg_abbr_from_mlb_name


def team_abbr(full_name: str | None) -> str:
    """Short code for scoreboard-style display (FG-style abbrev)."""
    if not full_name:
        return "?"
    return fg_abbr_from_mlb_name(full_name) or str(full_name)[:3].upper()


def is_final_row(row: pd.Series) -> bool:
    st = str(row.get("abstract_state") or "")
    det = str(row.get("status") or "")
    return st == "Final" or "Final" in det


def score_line_compact(row: pd.Series) -> str:
    """Away-first compact line, e.g. BOS17,BAL1."""
    ar, hr = row.get("away_runs"), row.get("home_runs")
    if ar is None or hr is None or (isinstance(ar, float) and math.isnan(ar)) or (isinstance(hr, float) and math.isnan(hr)):
        return "—"
    try:
        ai, hi = int(ar), int(hr)
    except (TypeError, ValueError):
        return "—"
    aa = team_abbr(row.get("away_name"))
    ha = team_abbr(row.get("home_name"))
    return f"{aa}{ai},{ha}{hi}"


def actual_winner_name(row: pd.Series) -> str:
    if not is_final_row(row):
        return "—"
    ar, hr = row.get("away_runs"), row.get("home_runs")
    if ar is None or hr is None or (isinstance(ar, float) and np.isnan(ar)) or (isinstance(hr, float) and np.isnan(hr)):
        return "—"
    try:
        ai, hi = int(ar), int(hr)
    except (TypeError, ValueError):
        return "—"
    if hi > ai:
        return str(row.get("home_name") or "—")
    if ai > hi:
        return str(row.get("away_name") or "—")
    return "Tie"
