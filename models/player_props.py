"""
Simple Bernoulli / Poisson-style prop estimates for key hitters vs probable opposing starter.

Uses MLB Stats API qualified team hitters and pitcher season lines (no FanGraphs).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.mlb_season_stats import league_average_pitcher_rates, pitcher_row_for_model, top_hitters_for_team_mlb
from utils.park_factors import park_factors_for_venue


def _per_game_rate(per_pa: float, pa: float = 4.2) -> float:
    """Expected count in a typical game plate appearances."""
    return float(np.clip(per_pa * pa, 0.05, 8.0))


def _pitcher_id(x: Any) -> int | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _pitcher_series(person_id: int | None, season: int) -> pd.Series:
    d = pitcher_row_for_model(person_id, season)
    d = {k: v for k, v in d.items() if k != "IP"}
    if not d:
        return pd.Series(league_average_pitcher_rates(season))
    return pd.Series(d)


def prop_row_for_hitter_vs_pitcher(
    hitter: pd.Series,
    pitcher_row: pd.Series,
    park_hr_factor: float,
    team_run_env: float,
    team_name: str,
) -> dict[str, Any]:
    """
    hitter: row with PA, H, HR, RBI, SO, Name (MLB-shaped).
    pitcher_row: opponent starter K/9, HR/9, etc.
    """
    pa = float(hitter.get("PA", 500) or 500)
    h = float(hitter.get("H", 0) or 0)
    hr = float(hitter.get("HR", 0) or 0)
    rbi = float(hitter.get("RBI", 0) or 0)
    so = float(hitter.get("SO", hitter.get("K", 0)) or 0)

    h_pa = h / max(pa, 1)
    hr_pa = hr / max(pa, 1)
    rbi_pa = rbi / max(pa, 1)
    k_pa = so / max(pa, 1)

    k9 = float(pitcher_row.get("K/9", 8.8) or 8.8)
    hr9 = float(pitcher_row.get("HR/9", 1.15) or 1.15)
    k_mult = float(np.clip(k9 / 8.8, 0.85, 1.15))
    hr_mult = float(np.clip(hr9 / 1.15, 0.85, 1.2))

    park_hr_mult = float(np.clip(park_hr_factor / 100.0, 0.85, 1.25))

    exp_hits = _per_game_rate(h_pa * 0.98)
    exp_hr = _per_game_rate(hr_pa * hr_mult * park_hr_mult * 1.05)
    exp_rbi = _per_game_rate(rbi_pa * (team_run_env / 4.35))
    exp_so = _per_game_rate(k_pa * k_mult * 1.05)

    return {
        "team": str(team_name or "—"),
        "player": str(hitter.get("Name", "")),
        "player_id": hitter.get("player_id"),
        "Pos": str(hitter.get("Pos") or ""),
        "exp_hits": round(exp_hits, 2),
        "exp_hr": round(exp_hr, 3),
        "exp_rbi": round(exp_rbi, 2),
        "exp_so": round(exp_so, 2),
    }


def pitcher_prop_display_row(
    team_name: str,
    side: str,
    pitcher_name: str | None,
    pitcher_id: int | None,
    season: int,
    opponent_exp_runs: float,
    actual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    One probable-starter row: coarse predicted game line + season anchors, optional actual boxscore line.
    ``opponent_exp_runs`` is the opposing team's expected runs (rough scale for predicted ER).
    ``team_name`` is the pitcher's club (Away or Home side in this matchup).
    """
    name = str(pitcher_name or "—")
    pr = pitcher_row_for_model(_pitcher_id(pitcher_id), season) if pitcher_id else {}
    pr = {k: v for k, v in pr.items() if k != "IP"}
    k9 = float(pr.get("K/9", 8.8) or 8.8)
    era = float(pr.get("ERA", 4.2) or 4.2)
    bb9 = float(pr.get("BB/9", 3.2) or 3.2)
    ip_hat = 5.5
    er_hat = max(0.25, float(opponent_exp_runs) * 0.42)
    h_hat = max(2.0, float(er_hat) * 1.35)
    k_hat = round((k9 / 9.0) * ip_hat, 1)
    bb_hat = round((bb9 / 9.0) * ip_hat, 1)
    act = actual or {}
    return {
        "Team": str(team_name or "—"),
        "Side": side,
        "Pitcher": name,
        "Pred IP": ip_hat,
        "Pred ER": round(er_hat, 2),
        "Pred H": round(h_hat, 1),
        "Pred BB": bb_hat,
        "Pred K": k_hat,
        "Season ERA": round(era, 2),
        "Season K/9": round(k9, 2),
        "Act IP": act.get("act_IP") or "—",
        "Act H": act.get("act_H", "—"),
        "Act R": act.get("act_R", "—"),
        "Act ER": act.get("act_ER", "—"),
        "Act BB": act.get("act_BB", "—"),
        "Act SO": act.get("act_SO", "—"),
    }


def props_for_game(
    season: int,
    home_team_id: int,
    away_team_id: int,
    home_team_name: str,
    away_team_name: str,
    home_fg: str,
    home_pitcher_id: int | None,
    away_pitcher_id: int | None,
    venue_id: int | None,
    home_exp_runs: float,
    away_exp_runs: float,
    top_n: int = 4,
) -> pd.DataFrame:
    hr_f, _ = park_factors_for_venue(venue_id, home_fg)
    away_p = _pitcher_series(_pitcher_id(away_pitcher_id), season)
    home_p = _pitcher_series(_pitcher_id(home_pitcher_id), season)

    rows: list[dict[str, float]] = []
    for _, h in top_hitters_for_team_mlb(int(home_team_id), season, top_n).iterrows():
        rows.append(prop_row_for_hitter_vs_pitcher(h, away_p, hr_f, home_exp_runs, home_team_name))
    for _, h in top_hitters_for_team_mlb(int(away_team_id), season, top_n).iterrows():
        rows.append(prop_row_for_hitter_vs_pitcher(h, home_p, hr_f, away_exp_runs, away_team_name))
    return pd.DataFrame(rows)
