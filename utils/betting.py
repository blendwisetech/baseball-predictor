"""Odds / Kelly helpers for exploratory bet-sizing UI (not financial advice)."""

from __future__ import annotations

import math
from typing import Any

from utils.betting_config import BettingPolicy, load_betting_policy


def american_to_implied_prob(american: float) -> float:
    """Vig-free implied win probability for one side from American odds."""
    if american is None or (isinstance(american, float) and math.isnan(american)):
        return float("nan")
    o = float(american)
    if o == 0.0:
        return float("nan")
    if o < 0:
        a = abs(o)
        return a / (a + 100.0)
    return 100.0 / (o + 100.0)


def american_to_decimal(american: float) -> float:
    """Decimal odds (stake + profit on a 1-unit win) from American."""
    o = float(american)
    if o < 0:
        return 1.0 + 100.0 / abs(o)
    return 1.0 + o / 100.0


def kelly_fraction(p_win: float, american: float) -> float:
    """
    Full Kelly fraction of bankroll for a single win bet at ``american`` odds.

    Uses net fractional odds b = D - 1 with decimal price D: f* = (p*b - (1-p)) / b.
    Returns 0 when the bet is not +EV at these odds.
    """
    if p_win is None or math.isnan(p_win) or p_win <= 0.0:
        return 0.0
    p_win = float(p_win)
    if p_win >= 1.0:
        p_win = 0.9999
    D = american_to_decimal(american)
    if D <= 1.001:
        return 0.0
    b = D - 1.0
    q = 1.0 - p_win
    num = p_win * b - q
    if num <= 0.0:
        return 0.0
    return num / b


def stake_skip_reason(model_p: float, implied_p: float, policy: BettingPolicy) -> str | None:
    """Human-readable reason when we refuse to size a bet."""
    if math.isnan(model_p) or math.isnan(implied_p):
        return "missing prob"
    if model_p < policy.min_pick_prob:
        return f"model < {policy.min_pick_prob:.0%}"
    edge_pct = (model_p - implied_p) * 100.0
    if edge_pct < policy.min_edge_pct:
        return f"edge < {policy.min_edge_pct:.1f}%"
    if edge_pct > policy.max_edge_pct:
        return f"edge > {policy.max_edge_pct:.1f}% (miscal?)"
    if kelly_fraction(model_p, -110) <= 0.0 and edge_pct <= 0:
        return "not +EV"
    return None


def suggest_stakes_quarter_kelly(
    bankroll: float,
    model_probs: list[float],
    american_odds: list[float],
    *,
    kelly_scale: float = 0.25,
    cap_per_bet: float = 0.15,
) -> list[float]:
    """
    Per-row dollar stakes: quarter-Kelly (by default), each row capped at ``cap_per_bet`` × bankroll,
    then scaled down uniformly if the sum exceeds ``bankroll``.
    """
    notes, stakes = suggest_stakes_with_policy(
        bankroll,
        model_probs,
        american_odds,
        BettingPolicy(kelly_scale=kelly_scale, cap_per_bet=cap_per_bet, max_slate_exposure=1.0),
    )
    return stakes


def suggest_stakes_with_policy(
    bankroll: float,
    model_probs: list[float],
    american_odds: list[float],
    policy: BettingPolicy | None = None,
    reg: dict[str, Any] | None = None,
) -> tuple[list[str], list[float]]:
    """
    Return per-row stake notes and dollar stakes.

    Applies min/max edge, minimum model probability, per-bet cap, and max slate exposure.
    """
    if policy is None:
        policy = load_betting_policy(reg)
    if bankroll <= 0.0:
        return ["no bankroll"] * len(model_probs), [0.0] * len(model_probs)

    notes: list[str] = []
    raw: list[float] = []
    for p, o in zip(model_probs, american_odds):
        impl = american_to_implied_prob(o)
        reason = stake_skip_reason(p, impl, policy)
        if reason:
            notes.append(reason)
            raw.append(0.0)
            continue
        k = kelly_fraction(p, o)
        stake = bankroll * policy.kelly_scale * max(0.0, k)
        stake = min(stake, bankroll * policy.cap_per_bet)
        if stake <= 0.0:
            notes.append("Kelly 0")
            raw.append(0.0)
        else:
            notes.append("sized")
            raw.append(stake)

    cap_total = bankroll * policy.max_slate_exposure
    total = sum(raw)
    if total > cap_total and total > 0.0:
        scale = cap_total / total
        raw = [s * scale for s in raw]
        notes = [("sized (slate cap)" if n == "sized" else n) for n in notes]
    return notes, raw
