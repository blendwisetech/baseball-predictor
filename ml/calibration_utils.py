"""
Post-hoc probability adjustments for the home-win model.

Isotonic mapping of tree ``predict_proba`` can sit on plateaus (many ≈0 / ≈1 scores).
Temperature scaling on logits spreads mass and typically improves Brier / log-loss
when fit on a held-out validation slice.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def apply_temperature(p: np.ndarray | float, T: float) -> np.ndarray | float:
    """
    ``p_new = sigmoid(logit(p) / T)``. ``T > 1`` pulls probabilities toward 0.5 (softer).
    ``T == 1`` is identity. Clamps ``p`` before ``logit`` for numerical safety.
    """
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        T = 1.0
    arr = np.asarray(p, dtype=float)
    flat = arr.ravel()
    if abs(T - 1.0) < 1e-12:
        out = np.clip(flat, 1e-6, 1.0 - 1e-6)
    else:
        lo = np.clip(flat, 1e-6, 1.0 - 1e-6)
        logit = np.log(lo / (1.0 - lo))
        z = logit / T
        exp_neg = np.exp(np.clip(-z, -50.0, 50.0))
        out = np.clip(1.0 / (1.0 + exp_neg), 1e-6, 1.0 - 1e-6)
    if arr.shape == ():
        return float(out[0])
    return out.reshape(arr.shape)


def tune_temperature_brier(p: np.ndarray, y: np.ndarray, *, T_hi: float = 4.0, n_grid: int = 61) -> float:
    """
    Pick ``T >= 1`` minimizing Brier score on validation labels ``y`` (0/1 away, 1 home).

    Search ``T in linspace(1, T_hi)``. Returns ``1.0`` if validation unusable.
    """
    p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=int).ravel()
    if len(p) < 8 or len(y) != len(p) or len(np.unique(y)) < 2:
        return 1.0
    Ts = np.linspace(1.0, float(T_hi), int(n_grid))
    best_t, best_b = 1.0, float("inf")
    for T in Ts:
        pt = np.asarray(apply_temperature(p, float(T)), dtype=float).ravel()
        pt = np.clip(pt, 1e-6, 1.0 - 1e-6)
        b = float(brier_score_loss(y, pt))
        if b < best_b:
            best_b, best_t = b, float(T)
    return best_t


def tune_marginal_shrink_brier(p: np.ndarray, y: np.ndarray, gamma: float, *, n_grid: int = 19) -> float:
    """
    Pick ``lam in [0, 0.45]`` minimizing Brier for ``p' = (1-lam)*p + lam*gamma``.

    ``gamma`` is usually the **training-only** home-win rate. ``lam=0`` is identity.
    """
    p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=int).ravel()
    gamma = float(np.clip(gamma, 0.35, 0.65))
    if len(p) < 8 or len(y) != len(p) or len(np.unique(y)) < 2:
        return 0.0
    best_lam, best_b = 0.0, float("inf")
    for lam in np.linspace(0.0, 0.45, int(n_grid)):
        lam = float(lam)
        pm = np.clip((1.0 - lam) * p + lam * gamma, 1e-6, 1.0 - 1e-6)
        b = float(brier_score_loss(y, pm))
        if b < best_b:
            best_b, best_lam = b, lam
    return float(best_lam)


def apply_marginal_shrink(p: np.ndarray | float, lam: float, gamma: float) -> np.ndarray | float:
    """Convex mix toward ``gamma``; ``lam=0`` leaves ``p`` unchanged."""
    if lam <= 0.0:
        return p
    gamma = float(np.clip(gamma, 0.35, 0.65))
    lam = float(np.clip(lam, 0.0, 0.49))
    arr = np.asarray(p, dtype=float)
    flat = arr.ravel()
    out = np.clip((1.0 - lam) * flat + lam * gamma, 1e-6, 1.0 - 1e-6)
    if arr.shape == ():
        return float(out[0])
    return out.reshape(arr.shape)


def shrink_toward_half(p: np.ndarray | float, strength: float) -> np.ndarray | float:
    """
    ``p' = 0.5 + strength * (p - 0.5)`` — pulls toward a **coin-flip** home model.

    ``strength`` in ``(0, 1]``: ``1`` = no change, ``0.75`` = retain 75% of the distance from 0.5
    (i.e. move 25% of the way toward 50%). Used to tame overconfident tails without retraining.
    """
    s = float(strength)
    if not np.isfinite(s) or s >= 0.9999 or s <= 0.0:
        return p
    s = float(np.clip(s, 1e-6, 1.0))
    arr = np.asarray(p, dtype=float)
    flat = arr.ravel()
    out = np.clip(0.5 + s * (flat - 0.5), 1e-6, 1.0 - 1e-6)
    if arr.shape == ():
        return float(out[0])
    return out.reshape(arr.shape)


def apply_symmetric_prob_cap(p: np.ndarray | float, half_width: float) -> np.ndarray | float:
    """Clip to ``[0.5 - half_width, 0.5 + half_width]`` (e.g. ``half_width=0.35`` → ``[0.15, 0.85]``)."""
    hw = float(half_width)
    if not np.isfinite(hw) or hw <= 0.0:
        return p
    hw = min(hw, 0.49)
    arr = np.asarray(p, dtype=float)
    flat = np.clip(arr.ravel(), 0.5 - hw, 0.5 + hw)
    if arr.shape == ():
        return float(flat[0])
    return flat.reshape(arr.shape)


def apply_registry_tail_calibration(p: np.ndarray | float, reg: dict | None) -> np.ndarray | float:
    """Apply ``production.win_prob_soften`` then optional ``production.win_prob_abs_cap``."""
    if reg is None:
        return p
    prod = reg.get("production") or {}
    s = float(prod.get("win_prob_soften", 1.0) or 1.0)
    out = shrink_toward_half(p, s)
    cap = float(prod.get("win_prob_abs_cap", 0.0) or 0.0)
    return apply_symmetric_prob_cap(out, cap)
