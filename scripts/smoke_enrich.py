"""Run from repo root: python scripts/smoke_enrich.py — writes scripts/smoke_enrich_out.json."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.feature_engineering import enrich_games_with_features
from utils.mlb_api import fetch_schedule, merge_schedule_with_probables

OUT = Path(__file__).resolve().parent / "smoke_enrich_out.json"


def main() -> None:
    picked = date.today()
    season = picked.year
    raw = fetch_schedule(picked, hydrate="probablePitcher(note),linescore(runners)")
    games = merge_schedule_with_probables(raw)
    if games.empty:
        picked = date(2024, 7, 4)
        season = 2024
        raw = fetch_schedule(picked, hydrate="probablePitcher(note),linescore(runners)")
        games = merge_schedule_with_probables(raw)
    if games.empty:
        OUT.write_text(json.dumps({"error": "no games"}), encoding="utf-8")
        return
    sub = games.head(2)
    enriched = enrich_games_with_features(sub, season, picked)
    ctx_cols = [
        c
        for c in enriched.columns
        if c.startswith("f_") and any(x in c for x in ("rest", "il", "venue", "pitch_depth", "prev_away"))
    ]
    rows = enriched[["gamePk", "home_name", "away_name"] + ctx_cols].to_dict(orient="records")
    OUT.write_text(json.dumps({"picked": str(picked), "n": len(enriched), "cols": ctx_cols, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
