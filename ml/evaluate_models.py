"""
Evaluate logged predictions and (when possible) replay the production win model
on merged rows with walk-forward style slices by calendar time.

Run: python -m ml.evaluate_models
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error

from ml.calibration_utils import apply_marginal_shrink, apply_registry_tail_calibration, apply_temperature
from ml.feature_config import GAME_FEATURE_NAMES, dataframe_X
from ml.time_split import resolve_sort_date, sort_by_time
from models.ml_predict import raw_home_win_prob_batch
from utils.data_io import PATH_MERGED, ROOT, ensure_dirs, load_registry, save_eval_report


def _win_prob_metrics_block(y: pd.Series, p: pd.Series) -> dict[str, Any]:
    yv = y.astype(int).values
    pv = p.astype(float).clip(1e-6, 1.0 - 1e-6).values
    return {
        "n": int(len(yv)),
        "brier": float(brier_score_loss(yv, pv)),
        "log_loss": float(log_loss(yv, pv, labels=[0, 1])),
        "accuracy_at_0.5": float(accuracy_score(yv, (pv >= 0.5).astype(int))),
    }


def _forward_window_logged_eval(df: pd.DataFrame) -> dict[str, Any] | None:
    """
    Logged win prob vs outcome, restricted to recent calendar windows by ``game_date``.
    Anchors use ``min(max(game_date), UTC today)`` so windows do not extend past wall-clock
    "today" when the merge file includes up-to-date games, while ``reference_end_max_game_date``
    records the newest game row either way.
    """
    if "home_win" not in df.columns or "pred_home_win_prob" not in df.columns:
        return None
    dts, col = resolve_sort_date(df)
    if col == "row_order":
        return None
    d = pd.to_datetime(dts, errors="coerce")
    if d.notna().sum() < 5:
        return None
    ref_raw = d.max()
    if pd.isna(ref_raw):
        return None
    ref_data = pd.Timestamp(pd.Timestamp(ref_raw).date())
    now_cap = pd.Timestamp(datetime.now(timezone.utc).date())
    try:
        ref_effective = min(ref_data, now_cap)
    except TypeError:
        ref_effective = ref_data
    y = df["home_win"].astype(int)
    p = df["pred_home_win_prob"].astype(float).clip(1e-6, 1 - 1e-6)
    out: dict[str, Any] = {
        "reference_end_max_game_date": str(ref_data.date()),
        "reference_end_effective": str(ref_effective.date()),
        "sort_column": col,
    }
    for days in (14, 30, 60):
        lo = ref_effective - pd.Timedelta(days=days - 1)
        m = d >= lo
        if int(m.sum()) < 3:
            continue
        out[f"last_{days}d"] = _win_prob_metrics_block(y.loc[m], p.loc[m])
    return out if len(out) > 2 else None


def _runs_win_consistency(df: pd.DataFrame) -> dict[str, Any] | None:
    """Agreement between run-side favorite and actual winner; correlation of win prob with margin."""
    need = ("home_win", "home_score", "away_score", "pred_home_runs", "pred_away_runs")
    if any(c not in df.columns for c in need):
        return None
    hs = pd.to_numeric(df["home_score"], errors="coerce")
    aws = pd.to_numeric(df["away_score"], errors="coerce")
    prh = pd.to_numeric(df["pred_home_runs"], errors="coerce")
    pra = pd.to_numeric(df["pred_away_runs"], errors="coerce")
    m = hs.notna() & aws.notna() & prh.notna() & pra.notna()
    if int(m.sum()) < 10:
        return None
    sub = df.loc[m]
    prh_s = sub["pred_home_runs"].astype(float)
    pra_s = sub["pred_away_runs"].astype(float)
    # Exclude predicted ties: ``>`` would label those as 0 (away), which skews agreement.
    run_tie = prh_s == pra_s
    sub2 = sub.loc[~run_tie]
    if len(sub2) < 10:
        return None
    run_fav_home = (sub2["pred_home_runs"].astype(float) > sub2["pred_away_runs"].astype(float)).astype(int).values
    hw = sub2["home_win"].astype(int).values
    agree = float((run_fav_home == hw).mean())
    out: dict[str, Any] = {
        "n": int(len(sub2)),
        "run_favorite_matches_winner": agree,
        "excluded_pred_run_ties": int(run_tie.sum()),
    }
    if "pred_home_win_prob" in sub2.columns:
        margin2 = (sub2["home_score"].astype(float) - sub2["away_score"].astype(float)).values
        ph = pd.to_numeric(sub2["pred_home_win_prob"], errors="coerce").astype(float).values
        ok = np.isfinite(ph) & np.isfinite(margin2)
        if int(ok.sum()) >= 10:
            rho = float(np.corrcoef(ph[ok], margin2[ok])[0, 1])
            if np.isfinite(rho):
                out["corr_pred_home_win_prob_with_margin"] = rho
    return out


def _rolling_time_metrics(df: pd.DataFrame, y_col: str, p_col: str, n_bins: int = 5) -> list[dict[str, Any]]:
    if df.empty or y_col not in df.columns or p_col not in df.columns:
        return []
    dts, _ = resolve_sort_date(df)
    sub = df[[y_col, p_col]].copy()
    sub["_dt"] = pd.to_datetime(dts, errors="coerce")
    sub = sub.dropna(subset=["_dt"])
    if len(sub) < n_bins * 3:
        return []
    try:
        sub["_bin"] = pd.qcut(sub["_dt"], q=n_bins, duplicates="drop")
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for name, grp in sub.groupby("_bin", observed=True):
        if len(grp) < 3:
            continue
        y = grp[y_col].astype(int)
        p = grp[p_col].astype(float).clip(1e-6, 1 - 1e-6)
        out.append(
            {
                "time_bin": str(name),
                "n": int(len(grp)),
                "brier": float(brier_score_loss(y, p)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "accuracy_at_0.5": float(accuracy_score(y, (p >= 0.5).astype(int))),
            }
        )
    return out


def _replay_win_model(df: pd.DataFrame, bundle: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any] | None:
    if "home_win" not in df.columns:
        return None
    df_s = sort_by_time(df.reset_index(drop=True))
    # Older merged rows may omit newer feature columns; ``dataframe_X`` pads with NaN.
    X = dataframe_X(df_s)
    y = df_s["home_win"].astype(int).values
    try:
        raw = raw_home_win_prob_batch(bundle, X)
    except Exception as e:
        return {"skipped": True, "reason": f"predict_error:{e}"}
    iso = bundle.get("iso")
    T = float(meta.get("win_temperature", meta.get("temperature", 1.0)) or 1.0)
    lam = float(meta.get("win_marginal_lambda", 0.0) or 0.0)
    gamma = float(meta.get("win_marginal_gamma", 0.535) or 0.535)
    if iso is not None:
        p_iso = np.clip(iso.predict(raw), 1e-6, 1.0 - 1e-6)
    else:
        p_iso = np.clip(raw, 1e-6, 1.0 - 1e-6)
    p_ml = np.asarray(apply_temperature(p_iso, T), dtype=float).ravel()
    p_ml = np.asarray(apply_marginal_shrink(p_ml, lam, gamma), dtype=float).ravel()
    p_ml = np.clip(p_ml, 1e-6, 1.0 - 1e-6)
    w = float(meta.get("blend_weight_val", 0.0) or 0.0)
    thresh = float(meta.get("home_threshold_val", 0.5) or 0.5)
    if "pred_home_win_prob_heur" in df_s.columns:
        ph = df_s["pred_home_win_prob_heur"].astype(float).values
        ph = np.clip(np.nan_to_num(ph, nan=0.5), 1e-6, 1.0 - 1e-6)
        p = (1.0 - w) * p_ml + w * ph
    else:
        p = p_ml
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    reg = load_registry()
    p = np.asarray(apply_registry_tail_calibration(p, reg), dtype=float).ravel()
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    rep_out: dict[str, Any] = {
        "n": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy_at_tuned_threshold": float(accuracy_score(y, (p >= thresh).astype(int))),
    }
    tmp = df_s.copy()
    tmp["_p_replay"] = p
    rep_out["rolling_replay_by_time"] = _rolling_time_metrics(tmp, "home_win", "_p_replay")
    return rep_out


def main() -> None:
    ensure_dirs()
    if not PATH_MERGED.exists():
        print("No merged data.")
        return
    df = pd.read_parquet(PATH_MERGED)
    report: dict[str, Any] = {"n_rows": len(df)}

    if "home_win" in df.columns and "pred_home_win_prob" in df.columns:
        y = df["home_win"].astype(int)
        p = df["pred_home_win_prob"].astype(float).clip(1e-6, 1 - 1e-6)
        report["win_logged"] = {
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "accuracy_at_0.5": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "rolling_by_time": _rolling_time_metrics(sort_by_time(df.copy()), "home_win", "pred_home_win_prob"),
        }
        fw = _forward_window_logged_eval(df)
        if fw:
            report["win_logged_forward_windows"] = fw

    rwc = _runs_win_consistency(df)
    if rwc:
        report["runs_win_consistency"] = rwc

    reg = load_registry()
    win_path = (reg.get("production") or {}).get("win_model_path")
    if win_path and (ROOT / win_path).exists():
        try:
            bundle = joblib.load(ROOT / win_path)
            meta = bundle.get("meta") or {}
            rep = _replay_win_model(df, bundle, meta)
            if rep:
                report["win_model_replay_on_merged"] = rep
        except Exception as e:
            report["win_model_replay_error"] = str(e)

    if all(c in df.columns for c in ("home_score", "away_score", "pred_home_runs", "pred_away_runs")):
        report["runs"] = {
            "mae_home": float(mean_absolute_error(df["home_score"], df["pred_home_runs"])),
            "mae_away": float(mean_absolute_error(df["away_score"], df["pred_away_runs"])),
            "mae_total": float(
                mean_absolute_error(
                    df["home_score"] + df["away_score"],
                    df["pred_home_runs"] + df["pred_away_runs"],
                )
            ),
        }
        if "pred_home_runs" in df.columns:
            report["runs"]["rolling_mae_total_by_time"] = []
            dts, _ = resolve_sort_date(df)
            tmp = df[["home_score", "away_score", "pred_home_runs", "pred_away_runs"]].copy()
            tmp["_dt"] = pd.to_datetime(dts, errors="coerce")
            tmp = tmp.dropna(subset=["_dt"])
            tmp["act_tot"] = tmp["home_score"] + tmp["away_score"]
            tmp["pred_tot"] = tmp["pred_home_runs"] + tmp["pred_away_runs"]
            if len(tmp) >= 15:
                try:
                    tmp["_bin"] = pd.qcut(tmp["_dt"], q=5, duplicates="drop")
                    for name, grp in tmp.groupby("_bin", observed=True):
                        if len(grp) < 3:
                            continue
                        report["runs"]["rolling_mae_total_by_time"].append(
                            {
                                "time_bin": str(name),
                                "n": int(len(grp)),
                                "mae_total": float(mean_absolute_error(grp["act_tot"], grp["pred_tot"])),
                            }
                        )
                except Exception:
                    pass

    save_eval_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
