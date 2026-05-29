"""
MLB moneylines from **Bovada.lv**’s public coupon JSON (no API key).

**Circa Sports (Nevada)** does not publish a documented public odds API. This module uses
Bovada’s unauthenticated ``/services/sports/event/coupon/.../baseball/mlb`` feed, which is
widely treated as a Vegas-style offshore reference (not affiliated with Circa).

If you later license Circa (e.g. OpticOdds), plug it in here as another ``provider``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import requests
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

BOVADA_MLB_COUPON_URL = (
    "https://www.bovada.lv/services/sports/event/coupon/events/A/description/baseball/mlb"
)


def _norm_team(s: str) -> str:
    return (s or "").strip().lower()


def _teams_exact(a1: str, h1: str, a2: str, h2: str) -> bool:
    return _norm_team(a1) == _norm_team(a2) and _norm_team(h1) == _norm_team(h2)


def _teams_loose(a1: str, h1: str, a2: str, h2: str) -> bool:
    def lo(x: str, y: str) -> bool:
        a, b = _norm_team(x), _norm_team(y)
        if not a or not b:
            return False
        if a == b:
            return True
        return len(a) >= 4 and (a in b or b in a)

    return lo(a1, a2) and lo(h1, h2)


def _parse_american(raw: Any) -> float | None:
    if raw is None:
        return None
    t = str(raw).strip().replace("\u2212", "-").replace("−", "-")
    try:
        return float(t)
    except ValueError:
        return None


def _event_et_date(start_ms: Any) -> date | None:
    if start_ms is None:
        return None
    try:
        ms = float(start_ms)
    except (TypeError, ValueError):
        return None
    dt = datetime.fromtimestamp(ms / 1000.0, tz=_ET)
    return dt.date()


def _competitor_names(ev: dict[str, Any]) -> tuple[str | None, str | None]:
    away_n = home_n = None
    for c in ev.get("competitors") or []:
        name = c.get("name")
        if not name:
            continue
        if c.get("home") is True:
            home_n = str(name)
        else:
            away_n = str(name)
    return away_n, home_n


def _pick_head_to_head_market(ev: dict[str, Any]) -> dict[str, Any] | None:
    """Prefer a non-live Head-to-Head (moneyline) market when both exist."""
    found: list[dict[str, Any]] = []
    for dg in ev.get("displayGroups") or []:
        for mkt in dg.get("markets") or []:
            if mkt.get("descriptionKey") != "Head To Head":
                continue
            outs = mkt.get("outcomes") or []
            if len(outs) != 2:
                continue
            found.append(mkt)
    if not found:
        return None
    non_live = [m for m in found if not (m.get("period") or {}).get("live")]
    pool = non_live if non_live else found
    pool.sort(key=lambda m: not (m.get("period") or {}).get("main", False))
    return pool[0]


def _moneylines_from_h2h(ev: dict[str, Any], mkt: dict[str, Any]) -> tuple[float | None, float | None]:
    id_to_side: dict[str, bool] = {}
    for c in ev.get("competitors") or []:
        cid = c.get("id")
        if cid is None:
            continue
        id_to_side[str(cid)] = bool(c.get("home"))

    away_ml = home_ml = None
    for o in mkt.get("outcomes") or []:
        cid = o.get("competitorId")
        price = _parse_american((o.get("price") or {}).get("american"))
        if cid is None or price is None:
            continue
        is_home = id_to_side.get(str(cid))
        if is_home is True:
            home_ml = price
        elif is_home is False:
            away_ml = price
    return away_ml, home_ml


def fetch_moneylines_for_slate_date(slate_date: date) -> list[dict[str, Any]]:
    """
    Pull moneylines for games whose **Eastern** start date equals ``slate_date``.

    Returns rows compatible with ``moneyline_for_pick`` in this module.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; baseball-predictor/1.0)"}
    r = requests.get(BOVADA_MLB_COUPON_URL, timeout=60, headers=headers)
    r.raise_for_status()
    payload = r.json()
    if not payload or not isinstance(payload, list):
        return []

    rows: list[dict[str, Any]] = []
    for league in payload:
        for ev in league.get("events") or []:
            ed = _event_et_date(ev.get("startTime"))
            if ed != slate_date:
                continue
            away_n, home_n = _competitor_names(ev)
            if not away_n or not home_n:
                continue
            mkt = _pick_head_to_head_market(ev)
            if mkt is None:
                continue
            aml, hml = _moneylines_from_h2h(ev, mkt)
            if aml is None or hml is None:
                continue
            rows.append(
                {
                    "source_id": str(ev.get("id") or ""),
                    "away_team": away_n,
                    "home_team": home_n,
                    "away_ml": float(aml),
                    "home_ml": float(hml),
                    "provider": "Bovada",
                }
            )
    return rows


def lookup_moneylines_pair(
    rows: list[dict[str, Any]],
    away_name: str,
    home_name: str,
) -> tuple[float | None, float | None]:
    for row in rows:
        a2, h2 = row.get("away_team") or "", row.get("home_team") or ""
        if _teams_exact(away_name, home_name, a2, h2):
            return row.get("away_ml"), row.get("home_ml")
    for row in rows:
        a2, h2 = row.get("away_team") or "", row.get("home_team") or ""
        if _teams_loose(away_name, home_name, a2, h2):
            return row.get("away_ml"), row.get("home_ml")
    return None, None


def moneyline_for_pick(
    rows: list[dict[str, Any]],
    away_name: str,
    home_name: str,
    pick_name: str,
) -> float | None:
    aml, hml = lookup_moneylines_pair(rows, away_name, home_name)
    if aml is None or hml is None:
        return None
    pn = _norm_team(pick_name)
    if pn == _norm_team(home_name):
        return float(hml)
    if pn == _norm_team(away_name):
        return float(aml)
    hn, an = _norm_team(home_name), _norm_team(away_name)
    if len(pn) >= 4 and (pn in hn or hn in pn):
        return float(hml)
    if len(pn) >= 4 and (pn in an or an in pn):
        return float(aml)
    return None
