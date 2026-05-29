"""
Simple preprocessing for tabular game features: median impute + scale.

ML: tree models are scale-invariant; linear / SGD benefit from scaling — we use Pipeline in trainers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def imputer_median() -> SimpleImputer:
    # Median is robust for a few missing pitcher lines early season.
    return SimpleImputer(strategy="median")


def scaler_pipeline() -> Pipeline:
    return Pipeline([("impute", imputer_median()), ("scale", StandardScaler())])


def fill_na_median(df: pd.DataFrame) -> pd.DataFrame:
    """In-place safe copy with column medians for quick evaluation outside sklearn."""
    out = df.copy()
    for c in out.columns:
        med = out[c].median()
        if np.isnan(med):
            out[c] = out[c].fillna(0.0)
        else:
            out[c] = out[c].fillna(med)
    return out
