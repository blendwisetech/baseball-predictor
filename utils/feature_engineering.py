"""
Build per-game feature rows from schedule + MLB Stats API season aggregates.

FanGraphs / pybaseball leaders often return HTTP 403 for automated clients; MLB JSON is reliable.

When ``features_as_of`` is True (default), team and starter lines use MLB ``byDateRange`` through the
calendar day before ``official_date`` (or the slate day), reducing end-of-season lookahead vs logging
historical games. If the YTD team table is empty, team stats fall back to full-season aggregates while
starters still use the as-of window when possible.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from utils.mlb_season_stats import (
    pitcher_row_for_model,
    team_hitting_mlb_table,
    team_hitting_mlb_table_through,
)
from utils.mlb_team_context import attach_high_impact_context
from utils.park_factors import park_factors_for_venue
from utils.team_map import fg_abbr_from_mlb_name


def _safe_int(x: Any) -> int | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _calendar_game_date(row: dict[str, Any], slate_date: date | None) -> date | None:
    od = row.get("official_date")
    if od is not None and str(od).strip():
        try:
            return date.fromisoformat(str(od)[:10])
        except ValueError:
            pass
    gd = row.get("gameDate")
    if gd is not None and str(gd).strip():
        try:
            return date.fromisoformat(str(gd)[:10])
        except ValueError:
            pass
    return slate_date


def _asof_stats_end(game_d: date | None, _season: int, slate_date: date | None) -> tuple[date | None, bool]:
    """
    Return (end_date_for_mlb_range, use_legacy_team_tables_only).

    ``use_legacy_team_tables_only`` is True when no calendar game day is known (cannot form an
    as-of cutoff). Stats are taken through the day before first pitch (inclusive end for MLB API).
    """
    if game_d is None:
        gd = slate_date
    else:
        gd = game_d
    if gd is None:
        return None, True
    asof = gd - timedelta(days=1)
    return asof, False


def enrich_games_with_features(
    games: pd.DataFrame,
    season: int,
    slate_date: date | None = None,
    *,
    features_as_of: bool = True,
) -> pd.DataFrame:
    """
    Input: merge_schedule_with_probables output.
    Joins team offense by MLB team id; probable starters by MLB people id.
    Keeps home_fg / away_fg abbreviations for park-factor lookup only.
    """
    if games.empty:
        return attach_high_impact_context(pd.DataFrame(), season, slate_date)

    use_asof = bool(features_as_of)
    team_cache: dict[str, pd.DataFrame] = {}
    legacy_off = team_hitting_mlb_table(season)
    if legacy_off.empty:
        legacy_idx = pd.DataFrame({"team_id": []}).astype({"team_id": "int64"}).set_index("team_id")
    else:
        legacy_idx = legacy_off.set_index("team_id")

    if use_asof:
        asof_keys: set[str] = set()
        for _, g in games.iterrows():
            rowd = g.to_dict()
            gd = _calendar_game_date(rowd, slate_date)
            asof, legacy = _asof_stats_end(gd, season, slate_date)
            if asof is not None and not legacy:
                asof_keys.add(asof.isoformat())
        for k in sorted(asof_keys):
            team_cache[k] = team_hitting_mlb_table_through(season, k)

    out_rows: list[dict[str, Any]] = []

    for _, g in games.iterrows():
        row = g.to_dict()
        row["home_fg"] = fg_abbr_from_mlb_name(g.get("home_name"))
        row["away_fg"] = fg_abbr_from_mlb_name(g.get("away_name"))

        gd = _calendar_game_date(row, slate_date)
        asof, legacy = _asof_stats_end(gd, season, slate_date)
        if not use_asof:
            legacy = True

        if legacy:
            off_idx = legacy_idx
            pitch_asof: date | None = None
        else:
            assert asof is not None
            tbl = team_cache.get(asof.isoformat())
            if tbl is None or tbl.empty:
                # Team YTD fetch can be empty early/API hiccup; still use as-of for starters.
                off_idx = legacy_idx
                pitch_asof = asof
            else:
                off_idx = tbl.set_index("team_id")
                pitch_asof = asof

        def attach_off(prefix: str, tid_raw: Any) -> None:
            tid = _safe_int(tid_raw)
            if tid is None or tid not in off_idx.index:
                return
            s = off_idx.loc[tid]
            pa = int(s["PA"]) if "PA" in s.index else 0
            row[f"{prefix}_team_PA"] = float(pa)
            for col in ["wRC+", "OBP", "SLG", "OPS", "BB%", "K%", "Barrel%", "Hard%", "wOBA"]:
                val = s[col] if col in s.index else np.nan
                if pd.notna(val) and np.isscalar(val):
                    row[f"{prefix}_team_{col}"] = float(val)

        attach_off("home", g.get("home_id"))
        attach_off("away", g.get("away_id"))

        hp = pitcher_row_for_model(_safe_int(g.get("home_probable_id")), season, through_end_date=pitch_asof)
        ap = pitcher_row_for_model(_safe_int(g.get("away_probable_id")), season, through_end_date=pitch_asof)
        for side, pr, label in (
            ("home", hp, "home_sp"),
            ("away", ap, "away_sp"),
        ):
            for k, v in pr.items():
                if k == "IP":
                    row[f"{side}_sp_IP"] = float(v)
                    continue
                row[f"{label}_{k}"] = v

        vid = _safe_int(g.get("venue_id"))
        abbr = str(row.get("home_fg") or "NYY")
        _hr_f, run_fac = park_factors_for_venue(vid, abbr)
        row["park_runs_factor"] = float(run_fac)

        out_rows.append(row)

    return attach_high_impact_context(pd.DataFrame(out_rows), season, slate_date)
