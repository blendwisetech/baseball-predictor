"""
Poisson-style expected runs from team offense (wRC+), starter suppression (FIP/xFIP),
and park run factor. Calibrated for personal use — tweak LEAGUE_AVG_RUNS with season context.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from utils.park_factors import park_factors_for_venue

# Rough league calibration (MLB ~4.2–4.8 runs/game per team depending on year)
LEAGUE_AVG_RUNS = 4.35
LEAGUE_AVG_FIP = 4.20


def _starter_run_suppression(fip: float | None, xfip: float | None) -> float:
    """
    Map pitcher FIP/xFIP to a multiplier on opponent runs (>1 means pitcher suppresses scoring).
    Uses average of FIP and xFIP when both present.
    """
    vals = [x for x in (fip, xfip) if x is not None and not math.isnan(float(x)) and float(x) > 0]
    if not vals:
        return 1.0
    ref = float(np.mean(vals))
    # Each run below league avg FIP trims ~6% from opponent expected runs (tunable)
    delta = (LEAGUE_AVG_FIP - ref) * 0.06
    return float(np.clip(1.0 + delta, 0.75, 1.35))


def _offense_multiplier(wrc_plus: float | None) -> float:
    if wrc_plus is None or math.isnan(float(wrc_plus)):
        return 1.0
    # wRC+ 110 -> +10% baseline runs vs average offense
    return float(np.clip(0.75 + (float(wrc_plus) / 100.0) * 0.25, 0.75, 1.35))


def expected_runs_one_side(
    team_wrc_plus: float | None,
    opp_starter_fip: float | None,
    opp_starter_xfip: float | None,
    park_runs_factor: float,
) -> float:
    """
    Expected runs for one team in one game (not a full distribution yet).
    Poisson lambda is this value rounded/truncated for display; simulation can sample Poisson(lambda).
    """
    lam = LEAGUE_AVG_RUNS
    lam *= _offense_multiplier(team_wrc_plus)
    lam *= _starter_run_suppression(opp_starter_fip, opp_starter_xfip)
    lam *= park_runs_factor / 100.0
    return float(np.clip(lam, 1.5, 9.0))


def project_game_runs(row: pd.Series) -> dict[str, float]:
    """
    row must include offensive wRC+ for home/away and starter FIP fields when available,
    plus venue_id, home_fg for park lookup.
    """
    hr_fac, run_fac = park_factors_for_venue(
        int(row["venue_id"]) if pd.notna(row.get("venue_id")) else None,
        str(row.get("home_fg") or "NYY"),
    )
    _ = hr_fac  # HR used in props; runs use run_fac

    home_runs_x = expected_runs_one_side(
        row.get("home_team_wRC+"),
        row.get("away_sp_FIP"),
        row.get("away_sp_xFIP"),
        run_fac,
    )
    away_runs_x = expected_runs_one_side(
        row.get("away_team_wRC+"),
        row.get("home_sp_FIP"),
        row.get("home_sp_xFIP"),
        run_fac,
    )
    return {"home_exp_runs": home_runs_x, "away_exp_runs": away_runs_x, "total_exp_runs": home_runs_x + away_runs_x}
