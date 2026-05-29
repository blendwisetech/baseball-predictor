"""
Backfill training data by logging historical predictions + outcomes for earlier dates.

This expands:
  - data/predictions/game_predictions.csv (features + predicted probs/runs)
  - data/outcomes/game_outcomes.csv (final scores)

Then `python -m ml.merge_training_data` can build a larger training parquet.

Notes / limitations:
  - Team and starter features default to MLB ``byDateRange`` through the day before each game's
    ``official_date`` (see ``utils.feature_engineering.enrich_games_with_features``). If the YTD
    team table is empty for that date, team offense falls back to full-season aggregates while
    starters still use the as-of window when the API returns a line.
  - Outcomes are pulled from MLB schedule linescore and only written when final.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import argparse

from ml.feature_config import GAME_FEATURE_NAMES, enriched_row_to_feature_vector
from models.ml_predict import load_production_pipelines, predict_runs_ml
from models.win_inference import resolve_game_win
from utils.data_io import append_predictions_df, upsert_outcomes, utc_now_iso
from utils.feature_engineering import enrich_games_with_features
from utils.mlb_api import fetch_schedule, merge_schedule_with_probables
from utils.mlb_outcomes import fetch_schedule_with_linescore, schedule_to_outcome_rows


@dataclass(frozen=True)
class BackfillResult:
    days: int
    games_seen: int
    preds_appended: int
    outcomes_upserted: int


def _daterange(start: date, end: date) -> list[date]:
    if end < start:
        return []
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d = d + timedelta(days=1)
    return out


def _existing_gamepks() -> set[int]:
    try:
        df = pd.read_csv("data/predictions/game_predictions.csv")
        if "gamePk" not in df.columns:
            return set()
        return set(pd.to_numeric(df["gamePk"], errors="coerce").dropna().astype(int).tolist())
    except Exception:
        return set()


def backfill_dates(start: date, end: date) -> BackfillResult:
    reg, win_b, runs_b = load_production_pipelines()
    existing = _existing_gamepks()

    days = 0
    games_seen = 0
    preds_appended = 0
    outcomes_upserted = 0

    for d in _daterange(start, end):
        days += 1
        season = int(d.year)

        # Outcomes (final scores) first
        try:
            raw_o = fetch_schedule_with_linescore(d)
            odf = schedule_to_outcome_rows(raw_o)
            if odf is not None and not odf.empty:
                outcomes_upserted += int(upsert_outcomes(odf))
        except Exception:
            # Best-effort; if MLB API hiccups for a day, continue.
            pass

        # Predictions log
        try:
            raw = fetch_schedule(d, hydrate="probablePitcher(note),linescore(runners)")
            games = merge_schedule_with_probables(raw)
        except Exception:
            continue

        if games is None or games.empty:
            continue

        games_seen += int(len(games))
        # Don't re-log games already in predictions file
        games = games[~games["gamePk"].astype(int).isin(existing)].copy()
        if games.empty:
            continue

        try:
            enriched = enrich_games_with_features(games, season, d)
        except Exception:
            continue

        log_rows: list[dict] = []
        for _, row in enriched.iterrows():
            pk = int(row["gamePk"])
            resolved = resolve_game_win(row, reg, win_b)
            proj_h = resolved["proj_h"]
            wp_h = resolved["wp_h"]
            ml_w = resolved["ml_w"]
            home_p = float(resolved["home_p"])
            away_p = float(resolved["away_p"])
            win_ver = resolved["win_ver"]
            ml_rh, ml_ra, runs_ver = predict_runs_ml(row, runs_b, reg)

            if ml_rh is not None and ml_ra is not None:
                ph, pa = float(ml_rh), float(ml_ra)
            else:
                ph, pa = float(proj_h["home_exp_runs"]), float(proj_h["away_exp_runs"])

            feats = enriched_row_to_feature_vector(row)
            log = {
                "logged_at": utc_now_iso(),
                "gamePk": pk,
                "game_date": d.isoformat(),
                "season": season,
                "home_id": row.get("home_id"),
                "away_id": row.get("away_id"),
                "home_name": row.get("home_name"),
                "away_name": row.get("away_name"),
                "pred_home_win_prob": home_p,
                "pred_away_win_prob": away_p,
                "pred_home_win_prob_heur": float(wp_h["home_win_prob"]),
                "pred_home_win_prob_ml": float(ml_w) if ml_w is not None else np.nan,
                "pred_home_runs": ph,
                "pred_away_runs": pa,
                "pred_home_runs_heur": float(proj_h["home_exp_runs"]),
                "pred_away_runs_heur": float(proj_h["away_exp_runs"]),
                "model_win_version": win_ver,
                "model_runs_version": runs_ver,
            }
            for k, v in feats.items():
                log[k] = v
            log_rows.append(log)
            existing.add(pk)

        if log_rows:
            log_df = pd.DataFrame(log_rows)
            # Ensure all expected feature columns exist, matching UI logging.
            for c in GAME_FEATURE_NAMES:
                if c not in log_df.columns:
                    log_df[c] = np.nan
            append_predictions_df(log_df)
            preds_appended += int(len(log_df))

    return BackfillResult(days=days, games_seen=games_seen, preds_appended=preds_appended, outcomes_upserted=outcomes_upserted)


def _opening_day_for_season(season: int) -> date:
    # Simple, safe default: March 15 (earlier than MLB opening day).
    # Days without games are skipped naturally.
    return date(season, 3, 15)


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill predictions/outcomes for earlier dates.")
    p.add_argument("--start", type=str, default="", help="Start date YYYY-MM-DD (default: season start).")
    p.add_argument("--end", type=str, default="", help="End date YYYY-MM-DD (default: yesterday).")
    args = p.parse_args()

    today = date.today()
    season = today.year
    start = _opening_day_for_season(season)
    end = today - timedelta(days=1)
    if args.start:
        start = date.fromisoformat(args.start)
    if args.end:
        end = date.fromisoformat(args.end)
    res = backfill_dates(start, end)
    print(
        {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days": res.days,
            "games_seen": res.games_seen,
            "preds_appended": res.preds_appended,
            "outcomes_upserted": res.outcomes_upserted,
        }
    )


if __name__ == "__main__":
    main()

