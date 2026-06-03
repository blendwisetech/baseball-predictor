"""
Load serialized sklearn models from data/models and score game rows.

Falls back to None when registry path missing or file absent — UI keeps heuristics.

Win bundle formats:
  - v2: ``base_pipeline`` + optional ``iso`` + ``meta.win_temperature`` + ``meta.win_marginal_lambda`` / ``win_marginal_gamma``
  - v1: single ``pipeline`` (no post-hoc calibration)
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
import pickle
import warnings

from ml.calibration_utils import apply_marginal_shrink, apply_registry_tail_calibration, apply_temperature
from ml.feature_config import dataframe_X, enriched_row_to_feature_vector
from utils.data_io import ROOT, load_registry


def _sklearn_X_matrix(X: pd.DataFrame) -> np.ndarray:
    """Column-ordered dense float matrix (no feature names) for older sklearn pipelines."""
    return np.ascontiguousarray(X.to_numpy(dtype=float))


def _bundle(rel_path: str | None) -> dict[str, Any] | None:
    """
    Load a joblib bundle from ``ROOT / rel_path``.

    Unpickling can fail on Streamlit Cloud when Python / numpy / sklearn do not match the
    environment that created the artifact (``ModuleNotFoundError`` inside ``pickle``). Return
    ``None`` so the app can fall back to heuristics instead of crashing.
    """
    if not rel_path:
        return None
    p = ROOT / rel_path
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except (
        ModuleNotFoundError,
        AttributeError,
        OSError,
        EOFError,
        ValueError,
        pickle.UnpicklingError,
    ) as e:
        warnings.warn(f"Skipping model bundle {p}: {type(e).__name__}: {e}", stacklevel=2)
        return None


def load_production_pipelines() -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Load registry once per app run; load joblib bundles at most twice (cheap for ~15 games)."""
    reg = load_registry()
    prod = reg.get("production", {})
    return reg, _bundle(prod.get("win_model_path")), _bundle(prod.get("runs_model_path"))


def raw_home_win_prob_batch(win_bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Uncalibrated P(home) for each row (same order as ``X``)."""
    Xv = _sklearn_X_matrix(X)
    if win_bundle.get("base_pipeline") is not None:
        return win_bundle["base_pipeline"].predict_proba(Xv)[:, 1]
    if win_bundle.get("pipeline") is not None:
        return win_bundle["pipeline"].predict_proba(Xv)[:, 1]
    raise KeyError("win bundle missing base_pipeline/pipeline")


def _win_prob_calibrated(raw: float, win_bundle: dict[str, Any], reg: dict[str, Any] | None = None) -> float:
    """Raw base ``predict_proba`` → isotonic (if any) → temperature → marginal shrink → optional ``win_prob_soften``."""
    meta = win_bundle.get("meta") or {}
    T = float(meta.get("win_temperature", meta.get("temperature", 1.0)) or 1.0)
    lam = float(meta.get("win_marginal_lambda", 0.0) or 0.0)
    gamma = float(meta.get("win_marginal_gamma", 0.535) or 0.535)
    r = float(np.clip(float(raw), 1e-6, 1.0 - 1e-6))
    iso = win_bundle.get("iso")
    if iso is None:
        p_iso = r
    else:
        p_iso = float(np.clip(iso.predict(np.array([r], dtype=float))[0], 1e-6, 1.0 - 1e-6))
    p_t = float(apply_temperature(p_iso, T))
    p = float(apply_marginal_shrink(p_t, lam, gamma))
    if reg is not None:
        p = float(apply_registry_tail_calibration(p, reg))
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))


def predict_home_win_ml(row: pd.Series, win_bundle: dict[str, Any] | None, reg: dict[str, Any]) -> tuple[float | None, str]:
    prod = reg.get("production", {})
    ver = str(prod.get("win_model_version", "heuristic_v1"))
    if not win_bundle:
        return None, ver
    if "base_pipeline" not in win_bundle and "pipeline" not in win_bundle:
        return None, ver
    feats = enriched_row_to_feature_vector(row)
    df1 = pd.DataFrame([feats])
    wmeta = win_bundle.get("meta") or {}
    feat_list = wmeta.get("features")
    if isinstance(feat_list, list) and feat_list:
        X = pd.DataFrame({c: df1[c] if c in df1.columns else np.nan for c in feat_list}).astype(float)
    else:
        X = dataframe_X(df1)
    try:
        raw = float(raw_home_win_prob_batch(win_bundle, X)[0])
    except Exception:
        return None, ver
    p = _win_prob_calibrated(raw, win_bundle, reg)
    return float(np.clip(p, 1e-6, 1.0 - 1e-6)), str(prod.get("win_model_version", ver))


def predict_runs_ml(
    row: pd.Series, runs_bundle: dict[str, Any] | None, reg: dict[str, Any]
) -> tuple[float | None, float | None, str]:
    prod = reg.get("production", {})
    ver = str(prod.get("runs_model_version", "heuristic_v1"))
    if not runs_bundle or "pipeline" not in runs_bundle:
        return None, None, ver
    feats = enriched_row_to_feature_vector(row)
    df1 = pd.DataFrame([feats])
    rmeta = runs_bundle.get("meta") or {}
    feat_list = rmeta.get("features")
    if isinstance(feat_list, list) and feat_list:
        X = pd.DataFrame({c: df1[c] if c in df1.columns else np.nan for c in feat_list}).astype(float)
    else:
        X = dataframe_X(df1)
    pred = runs_bundle["pipeline"].predict(_sklearn_X_matrix(X))[0]
    return float(pred[0]), float(pred[1]), str(prod.get("runs_model_version", ver))
