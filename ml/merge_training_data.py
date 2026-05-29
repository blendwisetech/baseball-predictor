"""
Join prediction log with outcome log by gamePk.

Produces data/processed/training_games.parquet for trainers.
Run: python -m ml.merge_training_data
"""

from __future__ import annotations

import numpy as np

from ml.feature_config import GAME_FEATURE_NAMES
from utils.data_io import PATH_MERGED, ROOT, ensure_dirs, load_outcomes, load_predictions


def main() -> None:
    ensure_dirs()
    pred = load_predictions()
    out = load_outcomes()
    if pred.empty or out.empty:
        print("Need non-empty predictions and outcomes CSVs before merge.")
        return
    if "gamePk" not in pred.columns or "gamePk" not in out.columns:
        print("gamePk column missing.")
        return
    out = out.drop_duplicates(subset=["gamePk"], keep="last")
    pred = pred.sort_values("logged_at").drop_duplicates(subset=["gamePk"], keep="last") if "logged_at" in pred.columns else pred.drop_duplicates(subset=["gamePk"], keep="last")
    merged = pred.merge(out, on="gamePk", how="inner", suffixes=("", "_out"))
    for c in GAME_FEATURE_NAMES:
        if c not in merged.columns:
            merged[c] = np.nan
    PATH_MERGED.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(PATH_MERGED, index=False)
    print(f"Wrote {len(merged)} rows to {PATH_MERGED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
