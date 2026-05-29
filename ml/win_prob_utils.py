"""Blend ML with heuristic win probability and tune a decision threshold on a validation set."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def tune_blend_weight(
    p_ml: np.ndarray,
    p_heur: np.ndarray | None,
    y: np.ndarray,
) -> float:
    """
    Pick blend weight w in [0, 1] minimizing Brier on y for
    p = (1-w)*p_ml + w*p_heur. If heur is unusable, returns 0.
    """
    p_ml = np.clip(np.asarray(p_ml, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y).astype(int)
    if p_heur is None or not np.isfinite(p_heur).any():
        return 0.0
    ph = np.clip(np.nan_to_num(np.asarray(p_heur, dtype=float), nan=0.5), 1e-6, 1.0 - 1e-6)
    best_w, best_b = 0.0, float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        p = (1.0 - w) * p_ml + w * ph
        b = float(brier_score_loss(y, p))
        if b < best_b:
            best_b, best_w = b, w
    return float(best_w)


def tune_home_threshold(p_blend: np.ndarray, y: np.ndarray) -> float:
    """Threshold on P(home win) maximizing hard-label accuracy vs y (0/1 away/home)."""
    p_blend = np.clip(np.asarray(p_blend, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y).astype(int)
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.33, 0.67, 35):
        acc = float((y == (p_blend >= t).astype(int)).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return float(best_t)


def blended_prob(p_ml: float, p_heur: float | None, w: float) -> float:
    p_ml = float(np.clip(p_ml, 1e-6, 1.0 - 1e-6))
    if p_heur is None or not np.isfinite(p_heur):
        return p_ml
    ph = float(np.clip(p_heur, 1e-6, 1.0 - 1e-6))
    w = float(np.clip(w, 0.0, 1.0))
    return float(np.clip((1.0 - w) * p_ml + w * ph, 1e-6, 1.0 - 1e-6))
