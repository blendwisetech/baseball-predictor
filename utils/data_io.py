"""
Load/save helpers for predictions, outcomes, registry, and training tables.

Append-only prediction logs support continuous learning without a database.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Project root = parent of `utils/`
ROOT = Path(__file__).resolve().parents[1]

PATH_PREDICTIONS = ROOT / "data" / "predictions" / "game_predictions.csv"
PATH_OUTCOMES = ROOT / "data" / "outcomes" / "game_outcomes.csv"
PATH_MERGED = ROOT / "data" / "processed" / "training_games.parquet"
PATH_REGISTRY = ROOT / "data" / "models" / "registry.json"
PATH_EVAL_REPORT = ROOT / "data" / "processed" / "eval_report.json"


def ensure_dirs() -> None:
    for p in (
        PATH_PREDICTIONS.parent,
        PATH_OUTCOMES.parent,
        PATH_MERGED.parent,
        PATH_REGISTRY.parent,
    ):
        p.mkdir(parents=True, exist_ok=True)


def load_registry() -> dict[str, Any]:
    ensure_dirs()
    if not PATH_REGISTRY.exists():
        return {}
    return json.loads(PATH_REGISTRY.read_text(encoding="utf-8"))


def save_registry(data: dict[str, Any]) -> None:
    ensure_dirs()
    PATH_REGISTRY.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_predictions_df(df: pd.DataFrame) -> None:
    """
    Append prediction rows. Uses concat + rewrite so new feature columns merge cleanly
    (fine for personal-scale CSV sizes).
    """
    ensure_dirs()
    if PATH_PREDICTIONS.exists() and PATH_PREDICTIONS.stat().st_size > 0:
        old = pd.read_csv(PATH_PREDICTIONS)
        combined = pd.concat([old, df], ignore_index=True, sort=False)
    else:
        combined = df
    combined.to_csv(PATH_PREDICTIONS, index=False)


def append_outcomes_df(df: pd.DataFrame) -> None:
    ensure_dirs()
    if PATH_OUTCOMES.exists() and PATH_OUTCOMES.stat().st_size > 0:
        df.to_csv(PATH_OUTCOMES, mode="a", header=False, index=False)
    else:
        df.to_csv(PATH_OUTCOMES, mode="w", header=True, index=False)


def load_predictions() -> pd.DataFrame:
    if not PATH_PREDICTIONS.exists():
        return pd.DataFrame()
    return pd.read_csv(PATH_PREDICTIONS)


def load_outcomes() -> pd.DataFrame:
    if not PATH_OUTCOMES.exists():
        return pd.DataFrame()
    return pd.read_csv(PATH_OUTCOMES)


def upsert_outcomes(df: pd.DataFrame) -> int:
    """Replace outcomes file with old + new rows, deduped by gamePk (latest wins)."""
    if df.empty:
        return 0
    ensure_dirs()
    old = load_outcomes()
    all_df = pd.concat([old, df], ignore_index=True) if not old.empty else df
    all_df = all_df.drop_duplicates(subset=["gamePk"], keep="last")
    all_df.to_csv(PATH_OUTCOMES, index=False)
    return len(df)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_eval_report(payload: dict[str, Any]) -> None:
    ensure_dirs()
    PATH_EVAL_REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_eval_report() -> dict[str, Any] | None:
    if not PATH_EVAL_REPORT.exists():
        return None
    return json.loads(PATH_EVAL_REPORT.read_text(encoding="utf-8"))
