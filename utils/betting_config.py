"""Bet-sizing policy loaded from ``registry.json``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BettingPolicy:
    """Guards against miscalibrated model edges blowing up Kelly stakes."""

    min_edge_pct: float = 2.0
    max_edge_pct: float = 10.0
    min_pick_prob: float = 0.52
    kelly_scale: float = 0.25
    cap_per_bet: float = 0.04
    max_slate_exposure: float = 0.20


def load_betting_policy(reg: dict[str, Any] | None) -> BettingPolicy:
    if not reg:
        return BettingPolicy()
    prod = reg.get("production") or {}
    raw = prod.get("betting") or reg.get("betting") or {}
    if not isinstance(raw, dict):
        return BettingPolicy()
    kw: dict[str, float] = {}
    for field in BettingPolicy.__dataclass_fields__:
        if field in raw and raw[field] is not None:
            try:
                kw[field] = float(raw[field])
            except (TypeError, ValueError):
                pass
    return BettingPolicy(**kw)
