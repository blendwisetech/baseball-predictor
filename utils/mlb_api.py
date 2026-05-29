"""
MLB Stats API helpers: schedule, probable pitchers, venue.
Public JSON — no key required. Suitable for personal-use tooling.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"


def fetch_schedule(game_date: date, hydrate: str | None = None) -> dict[str, Any]:
    """Return raw schedule JSON for MLB (sportId=1) on a calendar day."""
    d = game_date.isoformat()
    params: dict[str, str] = {"sportId": "1", "date": d}
    if hydrate:
        params["hydrate"] = hydrate
    r = requests.get(f"{MLB_STATS_BASE}/schedule", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def schedule_to_games_df(raw: dict[str, Any]) -> pd.DataFrame:
    """Flatten schedule JSON into one row per game with team ids/names and gamePk."""
    rows: list[dict[str, Any]] = []
    for day in raw.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            home = teams.get("home", {}).get("team", {})
            away = teams.get("away", {}).get("team", {})
            venue = g.get("venue", {})
            st = g.get("status") or {}
            rows.append(
                {
                    "gamePk": g.get("gamePk"),
                    "gameDate": g.get("gameDate"),
                    "official_date": g.get("officialDate")
                    or (str(g.get("gameDate"))[:10] if g.get("gameDate") else None),
                    "status": st.get("detailedState"),
                    "abstract_state": st.get("abstractGameState"),
                    "home_id": home.get("id"),
                    "home_name": home.get("name"),
                    "away_id": away.get("id"),
                    "away_name": away.get("name"),
                    "venue_id": venue.get("id"),
                    "venue_name": venue.get("name"),
                    "series_description": g.get("seriesDescription", ""),
                }
            )
    return pd.DataFrame(rows)


def attach_linescore_scores(raw: dict[str, Any]) -> pd.DataFrame:
    """Runs from embedded linescore when schedule hydrate includes linescore."""
    rows: list[dict[str, Any]] = []
    for day in raw.get("dates", []):
        for g in day.get("games", []):
            ls = g.get("linescore") or {}
            tc = ls.get("teams") or {}
            ar = (tc.get("away") or {}).get("runs")
            hr = (tc.get("home") or {}).get("runs")
            rows.append(
                {
                    "gamePk": g.get("gamePk"),
                    "away_runs": int(ar) if ar is not None else None,
                    "home_runs": int(hr) if hr is not None else None,
                }
            )
    return pd.DataFrame(rows)


def attach_probable_pitchers(raw: dict[str, Any]) -> pd.DataFrame:
    """
    Expect hydrate to include probablePitcher. Adds home/away probable id + fullName.
    """
    rows: list[dict[str, Any]] = []
    for day in raw.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            hp = teams.get("home", {}).get("probablePitcher") or {}
            ap = teams.get("away", {}).get("probablePitcher") or {}
            rows.append(
                {
                    "gamePk": g.get("gamePk"),
                    "home_probable_id": hp.get("id"),
                    "home_probable_name": hp.get("fullName"),
                    "away_probable_id": ap.get("id"),
                    "away_probable_name": ap.get("fullName"),
                }
            )
    return pd.DataFrame(rows)


def merge_schedule_with_probables(schedule_raw: dict[str, Any]) -> pd.DataFrame:
    base = schedule_to_games_df(schedule_raw)
    prob = attach_probable_pitchers(schedule_raw)
    if not prob.empty:
        base = base.merge(prob, on="gamePk", how="left")
    ls = attach_linescore_scores(schedule_raw)
    if not ls.empty:
        base = base.merge(ls, on="gamePk", how="left")
    return base


def cache_raw_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
