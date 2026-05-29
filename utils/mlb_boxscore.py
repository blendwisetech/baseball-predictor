"""
Boxscore batting / pitching lines for a completed game (MLB Stats API).

Used to attach actual game stats to prop tables. Returns empty frames when the
game is not final or the request fails.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import requests

from utils.mlb_api import MLB_STATS_BASE
from utils.mlb_season_stats import parse_innings_pitched


def _get(url: str) -> dict[str, Any]:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=384)
def _boxscore_json(game_pk: int) -> dict[str, Any] | None:
    try:
        return _get(f"{MLB_STATS_BASE}/game/{int(game_pk)}/boxscore")
    except Exception:
        return None


def batting_lines_game(game_pk: int) -> pd.DataFrame:
    """One row per player with ≥1 PA in the game."""
    raw = _boxscore_json(game_pk)
    if not raw:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for side in ("away", "home"):
        team_blk = (raw.get("teams") or {}).get(side) or {}
        tid = (team_blk.get("team") or {}).get("id")
        players = team_blk.get("players") or {}
        for _, p in players.items():
            stb = (p.get("stats") or {}).get("batting") or {}
            pa = int(stb.get("plateAppearances") or 0)
            if pa <= 0:
                continue
            person = p.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            pos = (p.get("position") or {}).get("abbreviation") or ""
            rows.append(
                {
                    "player_id": int(pid),
                    "act_AB": int(stb.get("atBats") or 0),
                    "act_H": int(stb.get("hits") or 0),
                    "act_HR": int(stb.get("homeRuns") or 0),
                    "act_RBI": int(stb.get("rbi") or 0),
                    "act_SO": int(stb.get("strikeOuts") or 0),
                    "act_R": int(stb.get("runs") or 0),
                    "act_BB": int(stb.get("baseOnBalls") or 0),
                    "batting_team_id": int(tid) if tid is not None else None,
                    "Pos_game": pos,
                }
            )
    return pd.DataFrame(rows)


def pitching_lines_game(game_pk: int) -> pd.DataFrame:
    """One row per pitcher who recorded outs in the game."""
    raw = _boxscore_json(game_pk)
    if not raw:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for side in ("away", "home"):
        team_blk = (raw.get("teams") or {}).get(side) or {}
        tid = (team_blk.get("team") or {}).get("id")
        players = team_blk.get("players") or {}
        for _, p in players.items():
            stp = (p.get("stats") or {}).get("pitching") or {}
            ip_txt = stp.get("inningsPitched")
            ip_num = parse_innings_pitched(ip_txt)
            if ip_num <= 0:
                continue
            person = p.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            rows.append(
                {
                    "player_id": int(pid),
                    "Pitcher": str(person.get("fullName") or ""),
                    "pitch_team_id": int(tid) if tid is not None else None,
                    "act_IP": str(ip_txt).strip() if ip_txt else "",
                    "act_IP_num": float(ip_num),
                    "act_H": int(stp.get("hits") or 0),
                    "act_R": int(stp.get("runs") or 0),
                    "act_ER": int(stp.get("earnedRuns") or 0),
                    "act_BB": int(stp.get("baseOnBalls") or 0),
                    "act_SO": int(stp.get("strikeOuts") or 0),
                    "act_BF": int(stp.get("battersFaced") or 0),
                }
            )
    return pd.DataFrame(rows)


def pitching_line_for_player(pitch_df: pd.DataFrame, person_id: int | None) -> dict[str, Any]:
    if person_id is None or pitch_df.empty or "player_id" not in pitch_df.columns:
        return {}
    sub = pitch_df.loc[pitch_df["player_id"] == int(person_id)]
    if sub.empty:
        return {}
    r = sub.iloc[0]
    return {
        "act_IP": r.get("act_IP", "—"),
        "act_H": int(r.get("act_H", 0)),
        "act_R": int(r.get("act_R", 0)),
        "act_ER": int(r.get("act_ER", 0)),
        "act_BB": int(r.get("act_BB", 0)),
        "act_SO": int(r.get("act_SO", 0)),
    }
