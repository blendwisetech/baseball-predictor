"""Smoke test for projection + props table logic. Writes scripts/smoke_ui_out.txt."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.components.slate_ui import actual_winner_name, is_final_row
from models.player_props import pitcher_prop_display_row, props_for_game
from utils.mlb_api import fetch_schedule, merge_schedule_with_probables
from utils.mlb_boxscore import batting_lines_game, pitching_line_for_player, pitching_lines_game


def _safe_pid(x):
    if x is None:
        return None
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def _decorate_hitter_props_for_display(pdf, game_pk: int):
    """Copy of app.main logic (avoids importing Streamlit app module)."""
    if pdf.empty:
        return pdf
    out = pdf.copy()
    if "player_id" in out.columns:
        out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    bat = batting_lines_game(int(game_pk))
    if not bat.empty and "player_id" in out.columns:
        b = bat.drop(columns=[c for c in ("Pos_game", "batting_team_id") if c in bat.columns], errors="ignore")
        out = out.merge(b, on="player_id", how="left")
    else:
        for c in ("act_AB", "act_H", "act_HR", "act_RBI", "act_SO", "act_R", "act_BB"):
            out[c] = np.nan
    disp_map = {
        "act_AB": "Act AB",
        "act_H": "Act H",
        "act_HR": "Act HR",
        "act_RBI": "Act RBI",
        "act_SO": "Act SO",
        "act_R": "Act R",
        "act_BB": "Act BB",
    }
    for k, label in disp_map.items():
        if k in out.columns:
            out[label] = out[k].apply(lambda v: "—" if pd.isna(v) else int(v))
        else:
            out[label] = "—"
    _drop = [c for c in list(disp_map.keys()) + ["player_id"] if c in out.columns]
    out = out.drop(columns=_drop, errors="ignore")
    out = out.rename(
        columns={
            "team": "Team",
            "player": "Player",
            "exp_hits": "Exp hits",
            "exp_hr": "Exp HR",
            "exp_rbi": "Exp RBI",
            "exp_so": "Exp SO",
        }
    )
    cols = [
        "Team",
        "Player",
        "Pos",
        "Exp hits",
        "Exp HR",
        "Exp RBI",
        "Exp SO",
        "Act AB",
        "Act H",
        "Act HR",
        "Act RBI",
        "Act SO",
        "Act R",
        "Act BB",
    ]
    return out[[c for c in cols if c in out.columns]]


def main() -> None:
    out_path = Path(__file__).resolve().parent / "smoke_ui_out.txt"
    out: dict = {}
    try:
        picked = date(2024, 7, 4)
        raw = fetch_schedule(picked, hydrate="probablePitcher(note),linescore(runners)")
        games = merge_schedule_with_probables(raw)
        out["games_n"] = len(games)
        if games.empty:
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            return
        row = games.iloc[0]
        pk = int(row["gamePk"])
        hid, aid = int(row["home_id"]), int(row["away_id"])
        ha = "NYY"
        pdf = props_for_game(
            season=2024,
            home_team_id=hid,
            away_team_id=aid,
            home_team_name=str(row.get("home_name") or "—"),
            away_team_name=str(row.get("away_name") or "—"),
            home_fg=ha,
            home_pitcher_id=row.get("home_probable_id"),
            away_pitcher_id=row.get("away_probable_id"),
            venue_id=None,
            home_exp_runs=4.2,
            away_exp_runs=4.1,
            top_n=2,
        )
        out["props_cols"] = list(pdf.columns)
        hit = _decorate_hitter_props_for_display(pdf, pk)
        out["hitter_cols"] = list(hit.columns)
        out["hitter_sample"] = hit.head(2).to_dict(orient="records")

        pitch_df = pitching_lines_game(pk)
        out["boxscore_pitch_rows"] = len(pitch_df)
        out["boxscore_bat_rows"] = len(batting_lines_game(pk))

        prow = []
        for side, pid, pname, ox, club in (
            ("Away", row.get("away_probable_id"), row.get("away_probable_name"), 4.2, str(row.get("away_name") or "—")),
            ("Home", row.get("home_probable_id"), row.get("home_probable_name"), 4.1, str(row.get("home_name") or "—")),
        ):
            act = pitching_line_for_player(pitch_df, _safe_pid(pid)) or None
            prow.append(
                pitcher_prop_display_row(
                    club,
                    side,
                    str(pname or ""),
                    _safe_pid(pid),
                    2024,
                    float(ox),
                    act,
                )
            )
        pit = pd.DataFrame(prow)
        out["pitcher_cols"] = list(pit.columns)
        out["pitcher_rows"] = pit.to_dict(orient="records")

        ar, hr = row.get("away_runs"), row.get("home_runs")
        out["is_final"] = bool(is_final_row(row))
        out["actual_winner"] = actual_winner_name(row)
        out["away_runs"] = None if ar is None or (isinstance(ar, float) and pd.isna(ar)) else int(ar)
        out["home_runs"] = None if hr is None or (isinstance(hr, float) and pd.isna(hr)) else int(hr)

        out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        out_path.write_text(json.dumps({"error": repr(e)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
