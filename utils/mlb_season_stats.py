"""
Season aggregates from the official MLB Stats API (no FanGraphs / no browser scraping).

FanGraphs often returns 403 to automated clients; MLB JSON is stable for personal tools.
"""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import requests

from utils.mlb_api import MLB_STATS_BASE

# Sample gates: shrink hot/cold early-season aggregates toward league anchors.
MIN_TEAM_PA_FOR_FULL_TRUST = 160
MIN_PITCHER_IP_FOR_FULL_TRUST = 14.0


def _get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    r = requests.get(url, params=params or {}, timeout=45)
    r.raise_for_status()
    return r.json()


def _first_split_stat(payload: dict[str, Any]) -> dict[str, Any] | None:
    stats = payload.get("stats") or []
    if not stats:
        return None
    splits = stats[0].get("splits") or []
    if not splits:
        return None
    return splits[0].get("stat") or None


def _best_pitching_split_stat(payload: dict[str, Any]) -> dict[str, Any] | None:
    """If a pitcher was traded, take the split with the most innings (simple merge proxy)."""
    stats = payload.get("stats") or []
    if not stats:
        return None
    splits = stats[0].get("splits") or []
    if not splits:
        return None
    best: dict[str, Any] | None = None
    best_ip = -1.0
    for sp in splits:
        st = sp.get("stat") or {}
        ip = parse_innings_pitched(st.get("inningsPitched"))
        if ip >= best_ip:
            best_ip = ip
            best = st
    return best


def _to_float(v: Any) -> float:
    if v is None or v == "":
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_innings_pitched(ip: Any) -> float:
    """
    MLB returns innings as '95.1' meaning 95 and one out (95 + 1/3).
    Whole number '95.0' is 95 innings.
    """
    if ip is None:
        return 0.0
    s = str(ip).strip()
    if not s:
        return 0.0
    m = re.match(r"^(\d+)(?:\.(\d))?$", s)
    if not m:
        return float(s)
    whole = int(m.group(1))
    frac = m.group(2)
    if not frac:
        return float(whole)
    outs = int(frac)
    if outs < 0 or outs > 2:
        return float(whole)
    return whole + outs / 3.0


def fip_from_counting(hr: float, bb: float, hbp: float, k: float, ip: float, c: float = 3.10) -> float:
    """Standard FIP with constant c (league-level; personal app default)."""
    if ip <= 0:
        return float("nan")
    return (13.0 * hr + 3.0 * (bb + hbp) - 2.0 * k) / ip + c


@lru_cache(maxsize=4)
def fetch_sport_team_ids(season: int) -> tuple[int, ...]:
    data = _get(f"{MLB_STATS_BASE}/teams", {"sportIds": 1, "season": season})
    teams = data.get("teams") or []
    return tuple(sorted(t["id"] for t in teams if t.get("id")))


@lru_cache(maxsize=8)
def team_hitting_mlb_table(season: int) -> pd.DataFrame:
    """
    One row per franchise: season team hitting from /teams/{id}/stats.
    wRC+ proxy = 100 * OPS / league_mean(OPS) (clipped) — not park-adjusted Sabermetric wRC+.
    """
    rows: list[dict[str, Any]] = []
    for tid in fetch_sport_team_ids(season):
        payload = _get(
            f"{MLB_STATS_BASE}/teams/{tid}/stats",
            {"stats": "season", "season": season, "group": "hitting", "gameType": "R"},
        )
        sp0 = ((payload.get("stats") or [{}])[0].get("splits") or [{}])[0]
        st = sp0.get("stat") or {}
        tm = sp0.get("team") or {}
        if not st:
            continue
        pa = int(st.get("plateAppearances") or 0)
        bb = float(st.get("baseOnBalls") or 0)
        so = float(st.get("strikeOuts") or 0)
        rows.append(
            {
                "team_id": tid,
                "team_name": tm.get("name"),
                "PA": pa,
                "OBP": _to_float(st.get("obp")),
                "SLG": _to_float(st.get("slg")),
                "OPS": _to_float(st.get("ops")),
                "BB%": 100.0 * bb / pa if pa else float("nan"),
                "K%": 100.0 * so / pa if pa else float("nan"),
                # Team counting totals for composite “lineup avg” props when no qualified hitters yet
                "team_hits": int(st.get("hits") or 0),
                "team_hr": int(st.get("homeRuns") or 0),
                "team_rbi": int(st.get("rbi") or 0),
                "team_so": int(st.get("strikeOuts") or 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    league_ops = float(np.nanmean(df["OPS"]))
    if not np.isfinite(league_ops) or league_ops <= 0:
        league_ops = 0.73
    df["wRC+"] = (100.0 * df["OPS"] / league_ops).clip(72.0, 132.0)
    # Placeholders so downstream column lists stay compatible with older FG-shaped frames
    df["Barrel%"] = np.nan
    df["Hard%"] = np.nan
    df["wOBA"] = np.nan
    return df


def _season_ytd_start_iso(season: int) -> str:
    """Inclusive start for YTD / byDateRange queries (before opening day returns sparse splits)."""
    return f"{season}-03-15"


@lru_cache(maxsize=512)
def team_hitting_mlb_table_through(season: int, end_date_iso: str) -> pd.DataFrame:
    """
    Franchise team hitting from ``start`` through ``end_date`` (inclusive), regular season only.
    Used for no-lookahead features at prediction time.
    """
    end_d = date.fromisoformat(end_date_iso)
    rows: list[dict[str, Any]] = []
    for tid in fetch_sport_team_ids(season):
        payload = _get(
            f"{MLB_STATS_BASE}/teams/{tid}/stats",
            {
                "stats": "byDateRange",
                "season": season,
                "group": "hitting",
                "gameType": "R",
                "startDate": _season_ytd_start_iso(season),
                "endDate": end_d.isoformat(),
            },
        )
        sp0 = ((payload.get("stats") or [{}])[0].get("splits") or [{}])[0]
        st = sp0.get("stat") or {}
        tm = sp0.get("team") or {}
        if not st:
            continue
        pa = int(st.get("plateAppearances") or 0)
        bb = float(st.get("baseOnBalls") or 0)
        so = float(st.get("strikeOuts") or 0)
        rows.append(
            {
                "team_id": tid,
                "team_name": tm.get("name"),
                "PA": pa,
                "OBP": _to_float(st.get("obp")),
                "SLG": _to_float(st.get("slg")),
                "OPS": _to_float(st.get("ops")),
                "BB%": 100.0 * bb / pa if pa else float("nan"),
                "K%": 100.0 * so / pa if pa else float("nan"),
                "team_hits": int(st.get("hits") or 0),
                "team_hr": int(st.get("homeRuns") or 0),
                "team_rbi": int(st.get("rbi") or 0),
                "team_so": int(st.get("strikeOuts") or 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    league_ops = float(np.nanmean(df["OPS"]))
    if not np.isfinite(league_ops) or league_ops <= 0:
        league_ops = 0.73
    # Blend team OPS (hence wRC+ proxy) toward league when PA is thin.
    pa = df["PA"].astype(float).clip(lower=0.0)
    alpha = (pa / float(MIN_TEAM_PA_FOR_FULL_TRUST)).clip(upper=1.0)
    ops_blend = (1.0 - alpha) * league_ops + alpha * df["OPS"].astype(float)
    df["OPS"] = ops_blend
    df["wRC+"] = (100.0 * df["OPS"] / league_ops).clip(72.0, 132.0)
    # OBP / SLG shrink toward league means when sample is small (keeps lineup shape sane).
    league_obp = float(np.nanmean(df["OBP"])) if df["OBP"].notna().any() else 0.32
    league_slg = float(np.nanmean(df["SLG"])) if df["SLG"].notna().any() else 0.41
    if not np.isfinite(league_obp):
        league_obp = 0.32
    if not np.isfinite(league_slg):
        league_slg = 0.41
    df["OBP"] = (1.0 - alpha) * league_obp + alpha * df["OBP"].astype(float)
    df["SLG"] = (1.0 - alpha) * league_slg + alpha * df["SLG"].astype(float)
    raw_bb = df["BB%"].astype(float).copy()
    raw_k = df["K%"].astype(float).copy()
    league_bb = float(np.nanmean(raw_bb)) if raw_bb.notna().any() else 8.5
    league_k = float(np.nanmean(raw_k)) if raw_k.notna().any() else 23.0
    if not np.isfinite(league_bb):
        league_bb = 8.5
    if not np.isfinite(league_k):
        league_k = 23.0
    df["BB%"] = (1.0 - alpha) * league_bb + alpha * raw_bb
    df["K%"] = (1.0 - alpha) * league_k + alpha * raw_k
    df["Barrel%"] = np.nan
    df["Hard%"] = np.nan
    df["wOBA"] = np.nan
    return df


@lru_cache(maxsize=8192)
def pitcher_pitching_stat_through(person_id: int, season: int, end_date_iso: str) -> dict[str, Any] | None:
    """Pitching line from season start through ``end_date`` (inclusive)."""
    payload = _get(
        f"{MLB_STATS_BASE}/people/{person_id}/stats",
        {
            "stats": "byDateRange",
            "season": season,
            "group": "pitching",
            "gameType": "R",
            "startDate": _season_ytd_start_iso(season),
            "endDate": end_date_iso,
        },
    )
    return _best_pitching_split_stat(payload)


@lru_cache(maxsize=512)
def pitcher_season_pitching_stat(person_id: int, season: int) -> dict[str, Any] | None:
    payload = _get(
        f"{MLB_STATS_BASE}/people/{person_id}/stats",
        {"stats": "season", "season": season, "group": "pitching", "gameType": "R"},
    )
    return _best_pitching_split_stat(payload)


def pitcher_row_for_model(
    person_id: int | None,
    season: int,
    *,
    through_end_date: date | None = None,
) -> dict[str, float]:
    """
    Numbers shaped like the old Fangraphs columns used in run_projection / props.

    When ``through_end_date`` is set, stats are MLB ``byDateRange`` through that calendar day
    (no same-game leakage when callers pass the day before first pitch). When unset, uses full
    season-to-date (current API behavior = rest-of-season lookahead vs a fixed past game).

    Includes ``IP`` (innings pitched in the window) for downstream confidence features; not a FG column.
    """
    if person_id is None:
        return {}
    pid = int(person_id)
    if through_end_date is not None:
        st = pitcher_pitching_stat_through(pid, season, through_end_date.isoformat())
    else:
        st = pitcher_season_pitching_stat(pid, season)
    if not st:
        return {}

    ip = parse_innings_pitched(st.get("inningsPitched"))
    hr = float(st.get("homeRuns") or 0)
    bb = float(st.get("baseOnBalls") or 0)
    hbp = float(st.get("hitByPitch") or st.get("hitBatsmen") or 0)
    k = float(st.get("strikeOuts") or 0)
    fip = fip_from_counting(hr, bb, hbp, k, ip)

    k9 = _to_float(st.get("strikeoutsPer9Inn"))
    bb9 = _to_float(st.get("walksPer9Inn"))
    hr9 = _to_float(st.get("homeRunsPer9"))
    era = _to_float(st.get("era"))

    anchors = league_average_pitcher_rates(season)
    alpha = min(1.0, float(ip) / float(MIN_PITCHER_IP_FOR_FULL_TRUST)) if ip > 0 else 0.0

    out: dict[str, float] = {"IP": float(ip)}
    if np.isfinite(era):
        out["ERA"] = float((1.0 - alpha) * anchors["ERA"] + alpha * era)
    if np.isfinite(fip):
        fip_b = float((1.0 - alpha) * anchors["FIP"] + alpha * float(fip))
        out["FIP"] = fip_b
        out["xFIP"] = fip_b
    if np.isfinite(k9):
        out["K/9"] = float((1.0 - alpha) * anchors["K/9"] + alpha * k9)
    if np.isfinite(bb9):
        out["BB/9"] = float((1.0 - alpha) * anchors["BB/9"] + alpha * bb9)
    if np.isfinite(hr9):
        out["HR/9"] = float((1.0 - alpha) * anchors["HR/9"] + alpha * hr9)
    babip = _to_float(st.get("avg"))
    if np.isfinite(babip):
        out["BABIP"] = float((1.0 - alpha) * anchors["BABIP"] + alpha * babip)
    return out


def _team_hitter_splits(team_id: int, season: int) -> list[dict[str, Any]]:
    """
    MLB defaults many clients to playerPool=QUALIFIED, which is often EMPTY early in April.
    Fall back to ALL, then filter pitchers and tiny samples.
    """
    base = {
        "stats": "season",
        "group": "hitting",
        "season": season,
        "teamId": team_id,
        "sportId": 1,
        "gameType": "R",
    }
    for pool in ("QUALIFIED", "ALL"):
        payload = _get(f"{MLB_STATS_BASE}/stats", {**base, "playerPool": pool})
        stats = payload.get("stats") or []
        if not stats:
            continue
        splits = stats[0].get("splits") or []
        if splits:
            return splits
    return []


def team_composite_batter_row(season: int, team_id: int) -> pd.Series:
    """
    One pseudo-row using full-team H/HR/RBI/SO per team PA (lineup-average rates).
    Used when there are still no individual hitter lines after QUALIFIED+ALL filtering.
    """
    tbl = team_hitting_mlb_table(season)
    if tbl.empty:
        return pd.Series(dtype=object)
    sub = tbl.loc[tbl["team_id"] == team_id]
    if sub.empty:
        return pd.Series(dtype=object)
    r = sub.iloc[0]
    pa = int(r.get("PA") or 0)
    if pa <= 0:
        return pd.Series(dtype=object)
    name = str(r.get("team_name") or f"Team {team_id}")
    return pd.Series(
        {
            "Name": f"{name} (team / PA)",
            "Team": team_id,
            "player_id": float("nan"),
            "Pos": "—",
            "PA": pa,
            "H": int(r.get("team_hits") or 0),
            "HR": int(r.get("team_hr") or 0),
            "RBI": int(r.get("team_rbi") or 0),
            "SO": int(r.get("team_so") or 0),
        }
    )


@lru_cache(maxsize=256)
def top_hitters_for_team_mlb(team_id: int, season: int, n: int, min_pa: int = 1) -> pd.DataFrame:
    """
    Top n hitters by PA for props. Tries QUALIFIED then ALL player pools.
    Drops pitchers and players below min_pa (default 1 so week-one samples still show; ALL includes pitchers as 0-PA hitters).
    """
    splits = _team_hitter_splits(team_id, season)
    rows: list[dict[str, Any]] = []
    for sp in splits:
        pos = (sp.get("position") or {}).get("abbreviation") or ""
        if pos == "P":
            continue
        st = sp.get("stat") or {}
        pl = sp.get("player") or {}
        tid = (sp.get("team") or {}).get("id")
        pa = int(st.get("plateAppearances") or 0)
        if pa < min_pa:
            continue
        rows.append(
            {
                "Name": pl.get("fullName", ""),
                "Team": tid,
                "player_id": pl.get("id"),
                "Pos": pos,
                "PA": pa,
                "H": int(st.get("hits") or 0),
                "HR": int(st.get("homeRuns") or 0),
                "RBI": int(st.get("rbi") or 0),
                "SO": int(st.get("strikeOuts") or 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        comp = team_composite_batter_row(season, team_id)
        if comp.empty:
            return pd.DataFrame()
        return pd.DataFrame([comp.to_dict()])
    df = df.sort_values("PA", ascending=False).head(n)
    return df


def league_average_pitcher_rates(_season: int) -> dict[str, float]:
    """League anchors when a pitcher has no MLB line yet (tunable by year if you want)."""
    return {"K/9": 8.8, "BB/9": 3.2, "HR/9": 1.15, "ERA": 4.20, "FIP": 4.20, "xFIP": 4.20, "BABIP": 0.290}
