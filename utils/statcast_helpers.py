"""
Optional Statcast wrappers via pybaseball. Keep calls lazy — Statcast can be slow.

Use for quality-of-contact features (xwOBA, barrel%) when you extend the model.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def statcast_batter_season(player_id: int, season: int) -> pd.DataFrame:
    """Batter-level Statcast for a season (may be empty if ID wrong or no data)."""
    try:
        from pybaseball import statcast_batter
    except ImportError:
        return pd.DataFrame()

    start = f"{season}-03-01"
    end = f"{season}-11-30"
    return statcast_batter(start, end, player_id)


def statcast_pitcher_season(player_id: int, season: int) -> pd.DataFrame:
    try:
        from pybaseball import statcast_pitcher
    except ImportError:
        return pd.DataFrame()

    start = f"{season}-03-01"
    end = f"{season}-11-30"
    return statcast_pitcher(start, end, player_id)


def summarize_statcast_contact(df: pd.DataFrame) -> dict[str, float]:
    """Derive simple contact-quality aggregates from a Statcast events frame."""
    if df is None or df.empty:
        return {
            "xwoba": 0.32,
            "barrel_rate": 0.07,
            "hard_hit_rate": 0.35,
            "avg_ev": 88.0,
        }
    out: dict[str, float] = {}
    if "estimated_woba_using_speedangle" in df.columns:
        out["xwoba"] = float(df["estimated_woba_using_speedangle"].mean())
    elif "xwoba" in df.columns:
        out["xwoba"] = float(df["xwoba"].mean())
    else:
        out["xwoba"] = 0.32

    bbe = df[df["description"].notna()] if "description" in df.columns else df
    if "launch_speed" in bbe.columns:
        out["avg_ev"] = float(bbe["launch_speed"].mean())
    else:
        out["avg_ev"] = 88.0

    if "launch_angle" in bbe.columns and "launch_speed" in bbe.columns:
        # crude proxy for barrels: EV>=95 and LA 10-50
        bar = (bbe["launch_speed"] >= 95) & (bbe["launch_angle"].between(10, 50))
        out["barrel_rate"] = float(bar.mean()) if len(bbe) else 0.07
        hh = bbe["launch_speed"] >= 95
        out["hard_hit_rate"] = float(hh.mean()) if len(bbe) else 0.35
    else:
        out["barrel_rate"] = 0.07
        out["hard_hit_rate"] = 0.35
    return out
