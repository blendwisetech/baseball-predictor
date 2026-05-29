"""
Lightweight column normalization for FanGraphs / pybaseball frames.
"""

from __future__ import annotations

import pandas as pd


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fg_team_key(name: str) -> str:
    """Normalize team label for joins (strip, upper)."""
    return str(name).strip().upper()
