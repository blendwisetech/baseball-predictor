"""
Calendar / season-progress helpers for model features (not true as-of stats).

Opening Day varies by year; we use a fixed late-March proxy so early-April games
get low ``f_season_day_norm`` and ``f_early_season_flag`` ≈ 1.
"""

from __future__ import annotations

from datetime import date

import numpy as np


def opening_day_proxy(season: int) -> date:
    """Conservative proxy for season start (MLB ~ late March)."""
    return date(int(season), 3, 22)


def season_progress_features(game_calendar_date: date | None, season: int) -> dict[str, float]:
    """
    Features merged into enriched rows and copied into ``GAME_FEATURE_NAMES``.

    ``f_season_day_norm``: ~0 at season proxy start, ~1 deep in summer (clip).
    ``f_early_season_flag``: 1.0 in first ~6 weeks of proxy calendar, else 0.
    """
    if game_calendar_date is None:
        return {"f_season_day_norm": float("nan"), "f_early_season_flag": float("nan")}
    try:
        gd = game_calendar_date if isinstance(game_calendar_date, date) else date.fromisoformat(str(game_calendar_date)[:10])
    except (TypeError, ValueError):
        return {"f_season_day_norm": float("nan"), "f_early_season_flag": float("nan")}
    open_d = opening_day_proxy(season)
    days = (gd - open_d).days
    # Playoff / next-year noise: clamp negative to 0
    days = max(0, int(days))
    # ~185 days regular season span for normalization
    f_season_day_norm = float(np.clip(days / 185.0, 0.0, 1.0))
    f_early_season_flag = 1.0 if days < 42 else 0.0
    return {"f_season_day_norm": f_season_day_norm, "f_early_season_flag": f_early_season_flag}
