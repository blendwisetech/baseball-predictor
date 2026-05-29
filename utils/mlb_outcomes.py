"""
Fetch final scores for completed games from MLB Stats API (linescore on schedule).
Used to build /data/outcomes for merging with predictions.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests

from utils.mlb_api import MLB_STATS_BASE


def fetch_schedule_with_linescore(game_date: date) -> dict[str, Any]:
    d = game_date.isoformat()
    params = {
        "sportId": "1",
        "date": d,
        "hydrate": "linescore(runners)",
    }
    r = requests.get(f"{MLB_STATS_BASE}/schedule", params=params, timeout=45)
    r.raise_for_status()
    return r.json()


def schedule_to_outcome_rows(raw: dict[str, Any]) -> pd.DataFrame:
    """
    One row per final game: gamePk, scores, home_win (1 if home won).
    Skips games without linescore or not Final.
    """
    rows: list[dict[str, Any]] = []
    for day in raw.get("dates", []):
        for g in day.get("games", []):
            st = (g.get("status") or {}).get("abstractGameState") or ""
            det = (g.get("status") or {}).get("detailedState") or ""
            if st != "Final" and "Final" not in det:
                continue
            ls = g.get("linescore") or {}
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            hruns = (ls.get("teams", {}).get("home", {}) or {}).get("runs")
            aruns = (ls.get("teams", {}).get("away", {}) or {}).get("runs")
            if hruns is None or aruns is None:
                continue
            hruns, aruns = int(hruns), int(aruns)
            rows.append(
                {
                    "gamePk": g.get("gamePk"),
                    "gameDate": g.get("gameDate"),
                    "home_id": (home.get("team") or {}).get("id"),
                    "away_id": (away.get("team") or {}).get("id"),
                    "home_name": (home.get("team") or {}).get("name"),
                    "away_name": (away.get("team") or {}).get("name"),
                    "home_score": hruns,
                    "away_score": aruns,
                    "home_win": int(hruns > aruns),
                    "total_runs": hruns + aruns,
                }
            )
    return pd.DataFrame(rows)
