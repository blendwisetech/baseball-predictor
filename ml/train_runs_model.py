"""
Train multi-output regression for home_score and away_score (actual runs).

Chronological split: report MAE on the most recent held-out slice; refit on train+val.

Run: python -m ml.train_runs_model
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from ml.feature_config import GAME_FEATURE_NAMES, dataframe_X
from ml.time_split import sort_by_time, time_train_val_test_masks
from utils.data_io import PATH_MERGED, ROOT, ensure_dirs, load_registry, save_registry


def _one_regressor(n_fit: int) -> Pipeline:
    kw: dict = dict(random_state=42, max_depth=5, max_iter=350, learning_rate=0.06, min_samples_leaf=12)
    if n_fit >= 120:
        kw["early_stopping"] = True
        kw["validation_fraction"] = 0.12
        kw["n_iter_no_change"] = 20
    else:
        kw["early_stopping"] = False
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("reg", HistGradientBoostingRegressor(**kw))])


def main() -> None:
    ensure_dirs()
    if not PATH_MERGED.exists():
        print("No merged training file.")
        return
    df0 = pd.read_parquet(PATH_MERGED)
    for c in ("home_score", "away_score"):
        if c not in df0.columns:
            print(f"Missing {c} in merged data.")
            return
    df = df0.dropna(subset=["home_score", "away_score"]).copy()
    df = sort_by_time(df)
    reg = load_registry()
    min_rows = int(reg.get("training", {}).get("min_rows_runs", 80))
    if len(df) < min_rows:
        print(f"Only {len(df)} rows; need >= {min_rows}.")
        return

    X = dataframe_X(df)
    y = df[["home_score", "away_score"]].astype(float)
    n = len(df)
    train_m, val_m, test_m = time_train_val_test_masks(n)
    tr_ix, va_ix, te_ix = np.flatnonzero(train_m), np.flatnonzero(val_m), np.flatnonzero(test_m)

    base_est = _one_regressor(len(tr_ix))
    mor = MultiOutputRegressor(base_est)
    mor.fit(X.iloc[tr_ix].values, y.iloc[tr_ix].values)
    pred_va = mor.predict(X.iloc[va_ix].values)
    mae_h = mean_absolute_error(y.iloc[va_ix].values[:, 0], pred_va[:, 0])
    mae_a = mean_absolute_error(y.iloc[va_ix].values[:, 1], pred_va[:, 1])

    fit_ix = np.flatnonzero(train_m | val_m)
    mor_final = MultiOutputRegressor(_one_regressor(len(fit_ix)))
    mor_final.fit(X.iloc[fit_ix].values, y.iloc[fit_ix].values)
    pred_te = mor_final.predict(X.iloc[te_ix].values)
    mae_te_h = mean_absolute_error(y.iloc[te_ix].values[:, 0], pred_te[:, 0])
    mae_te_a = mean_absolute_error(y.iloc[te_ix].values[:, 1], pred_te[:, 1])

    out_path = ROOT / "data" / "models" / "runs_mor.joblib"
    meta = {
        "type": "sklearn_multioutput_hgbr_runs",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "features": GAME_FEATURE_NAMES,
        "time_split": {"train": int(train_m.sum()), "val": int(val_m.sum()), "test": int(test_m.sum())},
        "mae_home_val": float(mae_h),
        "mae_away_val": float(mae_a),
        "mae_home_test": float(mae_te_h),
        "mae_away_test": float(mae_te_a),
    }
    joblib.dump({"pipeline": mor_final, "meta": meta}, out_path)
    print(json.dumps(meta, indent=2))

    reg.setdefault("production", {})
    reg["production"]["runs_model_path"] = str(out_path.relative_to(ROOT)).replace("\\", "/")
    reg["production"]["runs_model_version"] = f"mor_runs_{meta['trained_at'][:10]}"
    if "game_date" in df.columns:
        reg["production"]["trained_on_through"] = str(df["game_date"].max())
    save_registry(reg)


if __name__ == "__main__":
    main()
