"""
Shared win-probability and pick logic for the app, logging, and backfill.

Picks use the side with **higher** calibrated P(win), not a low home-only threshold.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.win_prob_utils import blended_prob
from models.ml_predict import predict_home_win_ml
from models.win_probability import win_probability_from_projection
from models.run_projection import project_game_runs


def blended_home_win_prob(reg: dict[str, Any], ml_w: float | None, wp: dict[str, float]) -> float:
    """Calibrated / blended P(home win) for display and logging."""
    prod = reg.get("production", {})
    w = float(prod.get("win_blend_weight", 0.0) or 0.0)
    p_heur = float(wp["home_win_prob"])
    if ml_w is None:
        return float(p_heur)
    return blended_prob(float(ml_w), p_heur, w)


def pick_winner_and_prob(
    home_name: str,
    away_name: str,
    home_p: float,
    away_p: float,
) -> tuple[str, float, str, float]:
    """Return (pick_name, pick_p, other_name, other_p)."""
    if float(home_p) >= float(away_p):
        return str(home_name), float(home_p), str(away_name), float(away_p)
    return str(away_name), float(away_p), str(home_name), float(home_p)


def resolve_game_win(
    row: pd.Series,
    reg: dict[str, Any],
    win_b: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    One enriched schedule row → runs heuristic, ML win prob, pick, and version metadata.
    """
    proj_h = project_game_runs(row)
    wp_h = win_probability_from_projection(proj_h)
    ml_w, win_ver = predict_home_win_ml(row, win_b, reg)
    home_p = blended_home_win_prob(reg, ml_w, wp_h)
    away_p = 1.0 - home_p
    pick_name, pick_p, _, _ = pick_winner_and_prob(
        str(row.get("home_name") or "—"),
        str(row.get("away_name") or "—"),
        home_p,
        away_p,
    )
    return {
        "proj_h": proj_h,
        "wp_h": wp_h,
        "ml_w": ml_w,
        "home_p": home_p,
        "away_p": away_p,
        "pick_name": pick_name,
        "pick_p": pick_p,
        "win_ver": win_ver,
    }
