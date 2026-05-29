"""
Win probability from projected run differential + home field.
Uses a logistic link (same family as logistic regression) with hand-tuned coefficients
for a transparent baseline. Swap in sklearn LogisticRegression when you have labeled training data.
"""

from __future__ import annotations

import math

import numpy as np


def logistic_win_prob_home(
    home_exp_runs: float,
    away_exp_runs: float,
    home_edge: float = 0.12,
    scale: float = 1.15,
) -> float:
    """
    P(home win) ~ sigmoid(scale * (log(home/away) + home_edge)).
    home_edge ~ empirical home-field in log-odds space (tunable).
    """
    ratio = (home_exp_runs + 1e-6) / (away_exp_runs + 1e-6)
    z = scale * (math.log(ratio) + home_edge)
    p = 1.0 / (1.0 + math.exp(-z))
    return float(np.clip(p, 0.02, 0.98))


def win_probability_from_projection(proj: dict[str, float]) -> dict[str, float]:
    ph = logistic_win_prob_home(proj["home_exp_runs"], proj["away_exp_runs"])
    return {"home_win_prob": ph, "away_win_prob": 1.0 - ph}
