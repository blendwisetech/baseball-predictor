"""
Chronological splits for game-level training (avoid future leakage).

Prefer `game_date` from prediction logs; fall back to `logged_at` when missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resolve_sort_date(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Return a datetime series aligned to df.index and the column name used.
    """
    if "game_date" in df.columns and df["game_date"].notna().any():
        s = pd.to_datetime(df["game_date"], errors="coerce")
        if s.notna().sum() >= max(1, int(0.5 * len(df))):
            return s, "game_date"
    if "logged_at" in df.columns:
        s = pd.to_datetime(df["logged_at"], errors="coerce")
        return s, "logged_at"
    return pd.Series(np.arange(len(df)), index=df.index), "row_order"


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Stable sort oldest → newest."""
    dates, _ = resolve_sort_date(df)
    out = df.copy()
    out["_sort_ts"] = dates
    out = out.sort_values("_sort_ts", kind="mergesort").drop(columns=["_sort_ts"])
    return out.reset_index(drop=True)


def time_train_val_test_masks(
    n: int,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contiguous boolean masks: train (oldest) | val | test (newest).
    Guarantees at least one row in each of val and test when n >= 3.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    train_m = np.zeros(n, dtype=bool)
    val_m = np.zeros(n, dtype=bool)
    test_m = np.zeros(n, dtype=bool)
    if n == 1:
        train_m[0] = True
        return train_m, val_m, test_m
    if n == 2:
        train_m[0] = True
        test_m[1] = True
        return train_m, val_m, test_m

    n_te = max(1, int(round(n * max(0.05, 1.0 - train_frac - val_frac))))
    n_va = max(1, int(round(n * val_frac)))
    n_tr = n - n_va - n_te
    if n_tr < 1:
        n_tr = 1
        n_va = max(1, (n - n_tr) // 2)
        n_te = n - n_tr - n_va
    train_m[:n_tr] = True
    val_m[n_tr : n_tr + n_va] = True
    test_m[n_tr + n_va :] = True
    return train_m, val_m, test_m
