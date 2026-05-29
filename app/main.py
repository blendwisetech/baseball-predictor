"""
Baseball Prediction App — Streamlit entrypoint.

Run from the `baseball_predictor` directory:
  pip install -r requirements.txt
  streamlit run app/main.py
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import math
import subprocess
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.components.slate_ui import actual_winner_name, is_final_row, score_line_compact
from ml.feature_config import GAME_FEATURE_NAMES, enriched_row_to_feature_vector
from models.ml_predict import load_production_pipelines, predict_runs_ml
from models.player_props import pitcher_prop_display_row, props_for_game
from models.run_projection import project_game_runs
from models.win_inference import resolve_game_win
from utils.betting import american_to_implied_prob, suggest_stakes_with_policy
from utils.betting_config import load_betting_policy
from utils.data_io import append_predictions_df, load_eval_report, upsert_outcomes, utc_now_iso
from utils.pipeline_runner import start_auto_pipeline_background
from utils.feature_engineering import enrich_games_with_features
from utils.mlb_api import fetch_schedule, merge_schedule_with_probables
from utils.mlb_outcomes import fetch_schedule_with_linescore, schedule_to_outcome_rows
from utils.mlb_boxscore import batting_lines_game, pitching_line_for_player, pitching_lines_game
from utils.scheduling import suggested_windows_text
from utils.nevada_odds import moneyline_for_pick

# Probables + linescore in one call so we can show results and status without extra clicks.
HYDRATE = "probablePitcher(note),linescore(runners)"

_PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def _format_game_start_pt(game_date_raw: object) -> str:
    """Pretty-print MLB ``gameDate`` (usually UTC ISO Z) as California local time."""
    if game_date_raw is None:
        return "—"
    try:
        if pd.isna(game_date_raw):
            return "—"
    except TypeError:
        pass

    try:
        if isinstance(game_date_raw, pd.Timestamp):
            dt = game_date_raw.to_pydatetime()
        elif isinstance(game_date_raw, datetime):
            dt = game_date_raw
        else:
            s = str(game_date_raw).strip()
            if not s:
                return "—"
            if s.endswith("Z"):
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        pt = dt.astimezone(_PACIFIC_TZ)
        h24 = pt.hour
        h12 = h24 % 12 or 12
        am_pm = "AM" if h24 < 12 else "PM"
        return f"{h12}:{pt.minute:02d} {am_pm} PT"
    except (ValueError, OSError, OverflowError):
        return "—"


@st.cache_data(ttl=180, show_spinner="Loading Nevada-area lines…")
def _cached_nevada_moneylines(slate_date: date, refresh_nonce: int) -> list[dict]:
    """``refresh_nonce`` busts cache when the user clicks **Refresh lines**."""
    from utils.nevada_odds import fetch_moneylines_for_slate_date

    return fetch_moneylines_for_slate_date(slate_date)


def _registry_mtime() -> float:
    """Bust pipeline cache when `registry.json` is rewritten after training."""
    p = _ROOT / "data" / "models" / "registry.json"
    try:
        return float(p.stat().st_mtime)
    except OSError:
        return 0.0


@st.cache_resource
def _cached_pipelines(_registry_mtime: float):
    return load_production_pipelines()


def get_pipelines():
    return _cached_pipelines(_registry_mtime())


def _win_model_loaded(win_b: dict | None) -> bool:
    return bool(win_b and (win_b.get("base_pipeline") is not None or win_b.get("pipeline") is not None))


def _runs_model_loaded(runs_b: dict | None) -> bool:
    return bool(runs_b and runs_b.get("pipeline") is not None)


def _show_table(df: pd.DataFrame, *, drop_cols: tuple[str, ...] = ()) -> None:
    if df.empty:
        st.caption("(no rows)")
        return
    d = df.drop(columns=list(drop_cols), errors="ignore")
    st.table(d.reset_index(drop=True))


def _model_prob_for_pick(row: pd.Series) -> float:
    """Model win probability for the picked side (fraction in (0,1)).

    Prefer full-precision ``pred_*_win_prob`` from projection rows. Rounded ``Home win %`` /
    ``Away win %`` alone can read as **100.0 → 1.0** and break Kelly (``p >= 1`` → zero stake).
    """
    pick = str(row.get("Pick") or "")
    home = str(row.get("Home") or "")
    away = str(row.get("Away") or "")
    if not pick or pick == "—":
        return float("nan")
    try:
        if pick == home:
            raw = row.get("pred_home_win_prob")
            if raw is not None and pd.notna(raw):
                return float(raw)
            return float(row["Home win %"]) / 100.0
        if pick == away:
            raw = row.get("pred_away_win_prob")
            if raw is not None and pd.notna(raw):
                return float(raw)
            return float(row["Away win %"]) / 100.0
    except (TypeError, ValueError, KeyError):
        return float("nan")
    return float("nan")


def _bet_sizing_section(pr: pd.DataFrame, slate_date: date) -> None:
    """Bankroll + Bovada (Vegas-style) moneylines → implied prob, edge, Kelly stakes."""
    if pr.empty or "gamePk" not in pr.columns:
        return

    st.subheader("Bet sizing — bankroll vs your odds")
    st.caption(
        "Set your **bankroll**. **American odds** load from **Bovada.lv**’s public MLB coupon JSON (no signup). "
        "**Circa Sports** does not publish a free developer odds feed; Bovada is a common Vegas-market reference "
        "(offshore, not Circa). Lines use the **Head-to-Head (moneyline)** market; we prefer pregame over live when both exist. "
        "Games **not** in the feed (name mismatch, wrong slate in ET, etc.) default to **−110** until you edit. "
        "**Pick** = team with higher model win %. Stakes use **¼ Kelly** with **edge** and **exposure** limits from "
        "`registry.json` → `production.betting` (skips huge edges that usually mean miscalibration). "
        "**Re-run projections** after changing calibration. Educational only—not betting advice."
    )

    c1, c2 = st.columns([1.4, 1.0])
    with c1:
        st.caption("Odds source: Bovada public coupon (Nevada-style reference; not Circa).")
    with c2:
        if st.button("Refresh lines", key="odds_refresh_btn", help="Bypass cache and refetch from Bovada."):
            st.session_state["odds_refresh_nonce"] = int(st.session_state.get("odds_refresh_nonce", 0)) + 1
            st.rerun()

    events: list[dict] = []
    odds_err: str | None = None
    try:
        nonce = int(st.session_state.get("odds_refresh_nonce", 0))
        events = _cached_nevada_moneylines(slate_date, nonce)
    except Exception as e:
        odds_err = str(e)
        events = []

    if odds_err:
        st.warning(f"Could not load Bovada lines: {odds_err}")

    bankroll = st.number_input(
        "Bankroll for this slate ($)",
        min_value=0.0,
        value=100.0,
        step=25.0,
        key="bet_bankroll_for_slate",
        help="Total dollars you want this table to allocate across the rows below.",
    )

    sig = "|".join(f'{int(r["gamePk"])}:{r.get("Pick")}' for _, r in pr.iterrows())
    sig_hash = hashlib.md5(sig.encode("utf-8")).hexdigest()[:12]

    default_ams: list[int] = []
    matched_live: list[bool] = []
    for _, r in pr.iterrows():
        home = str(r.get("Home") or "—")
        away = str(r.get("Away") or "—")
        pick = str(r.get("Pick") or "—")
        ao = moneyline_for_pick(events, away, home, pick) if events else None
        matched_live.append(ao is not None)
        default_ams.append(int(round(ao)) if ao is not None else -110)

    digest = hashlib.md5(str(tuple(default_ams)).encode("utf-8")).hexdigest()[:12]
    nonce = int(st.session_state.get("odds_refresh_nonce", 0))
    df_key = f"betsize_df_{sig_hash}_{digest}_n{nonce}"
    editor_key = f"betsize_ed_{sig_hash}_{digest}_n{nonce}"

    rows: list[dict] = []
    for (_, r), am in zip(pr.iterrows(), default_ams):
        pk = int(r["gamePk"])
        home = str(r.get("Home") or "—")
        away = str(r.get("Away") or "—")
        pick = str(r.get("Pick") or "—")
        mwp = _model_prob_for_pick(r)
        rows.append(
            {
                "gamePk": pk,
                "Matchup": f"{away} @ {home}",
                "Pick": pick,
                "Model win %": round(100 * mwp, 1) if not math.isnan(mwp) else float("nan"),
                "American odds (pick)": int(am),
            }
        )
    edit_src = pd.DataFrame(rows)

    if df_key not in st.session_state:
        st.session_state[df_key] = edit_src.copy()

    edited = st.data_editor(
        st.session_state[df_key],
        key=editor_key,
        hide_index=True,
        column_config={
            "gamePk": st.column_config.NumberColumn("gamePk", disabled=True, format="%d"),
            "Matchup": st.column_config.TextColumn("Matchup", disabled=True),
            "Pick": st.column_config.TextColumn("Pick", disabled=True),
            "Model win %": st.column_config.NumberColumn("Model win %", disabled=True, format="%.1f"),
            "American odds (pick)": st.column_config.NumberColumn(
                "American odds (pick)",
                help="Bovada moneyline on the **pick** when the slate matched; edit any cell.",
                step=1,
                format="%d",
            ),
        },
        use_container_width=True,
    )
    st.session_state[df_key] = edited

    if events:
        n_m = int(sum(matched_live))
        st.caption(
            f"Bovada moneylines matched for **{n_m}** of **{len(matched_live)}** picks "
            f"(slate **{slate_date.isoformat()}**, Eastern game date). "
            "Others stay at −110 when Bovada has no row or team names differ."
        )

    pr_by_pk = pr.drop_duplicates(subset=["gamePk"]).copy()
    pr_by_pk["_pk"] = pr_by_pk["gamePk"].astype(int)
    pr_by_pk = pr_by_pk.set_index("_pk")

    reg_bet, _, _ = get_pipelines()
    bet_policy = load_betting_policy(reg_bet)

    model_probs: list[float] = []
    american_list: list[float] = []
    out_rows: list[dict] = []

    for _, er in edited.iterrows():
        pk = int(er["gamePk"])
        try:
            r = pr_by_pk.loc[pk]
        except KeyError:
            model_probs.append(0.0)
            american_list.append(-110.0)
            out_rows.append(
                {
                    "Matchup": er.get("Matchup"),
                    "Pick": er.get("Pick"),
                    "American odds": -110,
                    "Implied win %": float("nan"),
                    "Model win %": float("nan"),
                    "Edge %": float("nan"),
                }
            )
            continue
        o_raw = float(er.get("American odds (pick)") or 0.0)
        o_eff = o_raw if o_raw != 0.0 else -110.0
        model_p = _model_prob_for_pick(r)
        impl = american_to_implied_prob(o_eff)
        if math.isnan(model_p) or math.isnan(impl):
            edge_pct = float("nan")
        else:
            edge_pct = (model_p - impl) * 100.0
        model_probs.append(model_p if not math.isnan(model_p) else 0.0)
        american_list.append(o_eff)

        out_rows.append(
            {
                "Matchup": er.get("Matchup"),
                "Pick": er.get("Pick"),
                "American odds": int(round(o_eff)),
                "Implied win %": round(100.0 * impl, 1) if not math.isnan(impl) else float("nan"),
                "Model win %": round(100.0 * model_p, 1) if not math.isnan(model_p) else float("nan"),
                "Edge %": round(edge_pct, 2) if not math.isnan(edge_pct) else float("nan"),
            }
        )

    stake_notes, stakes = suggest_stakes_with_policy(
        bankroll, model_probs, american_list, policy=bet_policy, reg=reg_bet
    )
    for i, stake in enumerate(stakes):
        if i < len(out_rows):
            out_rows[i]["Suggested stake ($)"] = round(stake, 2)
            out_rows[i]["Sizing"] = stake_notes[i] if i < len(stake_notes) else ""

    out_df = pd.DataFrame(out_rows)
    st.markdown("**Suggested allocation**")
    if out_df.empty:
        st.caption("No rows to size.")
        return
    _show_table(out_df)
    if bankroll > 0 and "Suggested stake ($)" in out_df.columns:
        total_st = float(out_df["Suggested stake ($)"].sum())
        st.caption(f"Total suggested stakes: **${total_st:.2f}** (bankroll entered: **${bankroll:.2f}**).")


def _run_cli_module(module: str) -> tuple[int, str, str]:
    r = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return r.returncode, r.stdout or "", r.stderr or ""


def _reload_slate(picked: date) -> pd.DataFrame:
    raw = fetch_schedule(picked, hydrate=HYDRATE)
    return merge_schedule_with_probables(raw)


def _sync_select_all() -> None:
    """Master checkbox: mirror to every row_* key for the current slate."""
    gdf = st.session_state.get("games_slate")
    if gdf is None or gdf.empty:
        return
    val = bool(st.session_state.get("cb_select_all", False))
    for pk in gdf["gamePk"].astype(int).tolist():
        st.session_state[f"row_{int(pk)}"] = val


def _slate_pick_signature(picked: date, games: pd.DataFrame) -> str:
    """Change when the calendar day or linescore rows change (so picks refresh after “Show outcomes”)."""
    parts: list[str] = []
    for _, r in games.sort_values("gamePk").iterrows():
        parts.append(f'{int(r["gamePk"])}:{r.get("away_runs")},{r.get("home_runs")}')
    return picked.isoformat() + "|" + "|".join(parts)


def _pick_accuracy_counts(games: pd.DataFrame, pred_pick: dict[int, str]) -> tuple[int, int]:
    """(correct, total) among final games that have both an actual winner and a predicted pick."""
    ok, n = 0, 0
    for _, row in games.iterrows():
        act = actual_winner_name(row)
        if act in ("—", "Tie"):
            continue
        pk = int(row["gamePk"])
        pred = pred_pick.get(pk, "—")
        if pred == "—":
            continue
        n += 1
        if str(pred) == str(act):
            ok += 1
    return ok, n


def _format_pick_cell(row: pd.Series, pred_pick: dict[int, str], pk: int) -> str:
    pred = pred_pick.get(pk, "—")
    act = actual_winner_name(row)
    if pred == "—":
        return html.escape("—")
    esc = html.escape(str(pred))
    if act in ("—", "Tie"):
        return esc
    if str(pred) == str(act):
        return f'<span style="color:#1a7f37;font-weight:600">{esc}</span>'
    return f'<span style="color:#c41e3a;font-weight:600">{esc}</span>'


def _safe_pid(x) -> int | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _decorate_hitter_props_for_display(pdf: pd.DataFrame, game_pk: int) -> pd.DataFrame:
    """Merge boxscore batting lines; drop raw ids; human-readable column names."""
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


def _compute_win_pick_per_game(games: pd.DataFrame, season: int, slate_date: date | None = None) -> dict[int, str]:
    """One predicted winner per gamePk — team with higher calibrated P(win)."""
    reg, win_b, _ = get_pipelines()
    enriched = enrich_games_with_features(games, season, slate_date)
    out: dict[int, str] = {}
    for _, row in enriched.iterrows():
        pk = int(row["gamePk"])
        resolved = resolve_game_win(row, reg, win_b)
        out[pk] = resolved["pick_name"]
    return out


def main() -> None:
    if os.environ.get("SPORTS_HUB") != "1":
        st.set_page_config(page_title="Baseball Predictor", layout="wide")
    st.title("Baseball game predictions")
    st.caption(
        "**Pick (predicted)** is the team with higher calibrated win % (ML + optional heuristic blend when trained). "
        "Use **Run projections for selected** for full run lines, props, and CSV logging. "
        "**Browser:** blank page or `static/css` / `Button.*.js` errors → Chrome/Edge, allow localhost in blockers, "
        "hard-refresh (Ctrl+Shift+R), `pip install -U streamlit` (≥1.36)."
    )

    today = date.today()
    picked = st.date_input("Game day", value=today, min_value=date(2020, 1, 1))
    season = picked.year

    with st.sidebar:
        maint_ok = st.session_state.pop("_maint_ok_message", None)
        if maint_ok:
            st.success(maint_ok)
        st.subheader("Automation")
        st.checkbox(
            "Auto merge / train / eval after logging projections or ingesting outcomes",
            value=True,
            key="auto_pipeline_bg",
            help="Runs in the background so the UI stays responsive. Check data/processed/auto_pipeline.log.",
        )
        st.header("Models")
        reg, win_b, runs_b = get_pipelines()
        prod = reg.get("production", {})
        w_ok, r_ok = _win_model_loaded(win_b), _runs_model_loaded(runs_b)
        if w_ok and r_ok:
            st.success("ML bundles loaded (win + runs). Projections will use them.")
        elif w_ok or r_ok:
            st.warning(
                "Only some ML bundles loaded — "
                + ("win OK · " if w_ok else "win missing · ")
                + ("runs OK" if r_ok else "runs missing")
                + ". Run **Full ML bootstrap** below after merge, or train individually."
            )
        else:
            st.info(
                "Using **heuristic_v1** for win/runs (no trained joblib in registry, or paths missing). "
                "Log projections for games with final scores, **Ingest outcomes**, then **Full ML bootstrap**."
            )
        st.write(f"**Win:** `{prod.get('win_model_version', '—')}`")
        st.write(f"**Runs:** `{prod.get('runs_model_version', '—')}`")
        st.write(f"**Props:** `{prod.get('props_model_version', '—')}`")
        if prod.get("win_model_path"):
            st.caption(f"path: `{prod.get('win_model_path')}`")
        bp = load_betting_policy(reg)
        st.caption(
            f"Win: blend **{prod.get('win_blend_weight', 0.0)}** · T **{prod.get('win_temperature', 1.0)}** · "
            f"λ **{prod.get('win_marginal_lambda', 0.0)}** · soften **{prod.get('win_prob_soften', 1.0)}** · "
            f"cap ±**{prod.get('win_prob_abs_cap', 0.0)}**"
        )
        st.caption(
            f"Bet sizing: edge **{bp.min_edge_pct:.1f}–{bp.max_edge_pct:.1f}%** · "
            f"min pick **{bp.min_pick_prob:.0%}** · Kelly **{bp.kelly_scale:.0%}** · "
            f"max **{bp.cap_per_bet:.0%}**/game · slate cap **{bp.max_slate_exposure:.0%}**"
        )
        st.divider()
        st.subheader("Recent evaluation")
        ev = load_eval_report()
        if ev:
            with st.expander("eval_report.json (truncated)", expanded=False):
                txt = json.dumps(ev, indent=2, default=str)
                st.code(txt[:14000] + ("\n… (truncated)" if len(txt) > 14000 else ""), language="json")
        else:
            st.caption("Run **Evaluate models** (maintenance form) after merge to populate `eval_report.json`.")
        st.divider()
        st.markdown(suggested_windows_text())
        st.subheader("Outcomes file")
        if st.button(
            f"Ingest final scores for {picked.isoformat()}",
            key="ingest_outcomes_btn",
        ):
            try:
                raw = fetch_schedule_with_linescore(picked)
                odf = schedule_to_outcome_rows(raw)
                n = upsert_outcomes(odf)
                st.success(f"Ingested {n} rows into outcomes CSV.")
                if st.session_state.get("auto_pipeline_bg", True) and n > 0:
                    start_auto_pipeline_background()
                    st.caption("Started background merge / train / eval.")
            except Exception as e:
                st.error(str(e))
        st.subheader("Maintenance (local)")
        _maint_actions = {
            "Merge predictions + outcomes": "ml.merge_training_data",
            "Train win model": "ml.train_win_model",
            "Train runs model": "ml.train_runs_model",
            "Evaluate models": "ml.evaluate_models",
            "Full ML bootstrap (merge + train win + train runs + eval)": "ml.bootstrap_models",
        }
        with st.form("sidebar_maint_form"):
            maint_choice = st.selectbox(
                "Choose one action",
                list(_maint_actions.keys()),
                key="sidebar_maint_choice",
            )
            run_maint = st.form_submit_button("Run selected action")
        if run_maint:
            mod = _maint_actions[maint_choice]
            code, out, err = _run_cli_module(mod)
            block = (out + "\n" + err).strip() or f"(no output, exit {code})"
            st.code(block[-20000:], language="text")
            if code == 0:
                st.session_state["_maint_ok_message"] = f"{maint_choice} completed."
                st.rerun()
            else:
                st.error(f"Exit code {code}. Fix errors above, then retry.")

    # Load / refresh slate when the calendar day changes
    if "games_slate" not in st.session_state or st.session_state.get("_picked") != picked:
        st.session_state.games_slate = _reload_slate(picked)
        st.session_state._picked = picked
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and k.startswith("row_"):
                del st.session_state[k]
        st.session_state.cb_select_all = False
        st.session_state.pred_pick = {}
        st.session_state._win_pick_sig = None
        st.session_state.has_proj = False
        st.session_state.pop("proj_results", None)
        st.session_state.pop("proj_props_concat", None)
        st.session_state.pop("proj_hitters_props", None)
        st.session_state.pop("proj_pitchers_props", None)

    games = st.session_state.games_slate
    if games.empty:
        st.warning("No MLB games on that date (or schedule not published yet).")
        return

    pick_sig = _slate_pick_signature(picked, games)
    if st.session_state.get("_win_pick_sig") != pick_sig:
        with st.spinner("Predicting winner for each game…"):
            try:
                st.session_state.pred_pick = _compute_win_pick_per_game(games, season, picked)
                st.session_state.pop("_pick_error", None)
            except Exception as e:
                st.session_state.pred_pick = {}
                st.session_state["_pick_error"] = str(e)
        st.session_state._win_pick_sig = pick_sig
    if st.session_state.get("_pick_error"):
        st.warning(f"Could not auto-fill picks: {st.session_state['_pick_error']}")

    st.subheader(f"Games on {picked.isoformat()}")

    pred_pick = st.session_state.get("pred_pick", {})
    acc_ok, acc_n = _pick_accuracy_counts(games, pred_pick)

    # First column must fit "Select all games" beside the checkbox (narrow 0.45 caused vertical label wrap).
    _slate_cols = [1.05, 1.02, 1.02, 0.88, 1.08, 1.22, 1.28, 1.38]
    h = st.columns(_slate_cols)
    h[0].markdown("")
    h[1].markdown("**Away**")
    h[2].markdown("**Home**")
    h[3].markdown("**Start (PT)**")
    h[4].markdown("**Status**")
    h[5].markdown("**Result**")
    h[6].markdown("**Winner (actual)**")
    with h[7]:
        if acc_n > 0:
            pct = 100.0 * acc_ok / acc_n
            st.caption(f"**{pct:.0f}%** correct ({acc_ok}/{acc_n} final)")
        else:
            st.caption("_Pick accuracy: no graded games yet_")
        st.markdown("**Pick (predicted)**")

    sel = st.columns(_slate_cols)
    with sel[0]:
        st.checkbox(
            "Select all games",
            key="cb_select_all",
            on_change=_sync_select_all,
            help="Sets every row checkbox on or off.",
        )

    for _, row in games.iterrows():
        pk = int(row["gamePk"])
        c0, c1, c2, c3, c4, c5, c6, c7 = st.columns(_slate_cols)
        with c0:
            st.checkbox("sel", key=f"row_{pk}", label_visibility="collapsed")
        with c1:
            nm = row.get("away_name") or "—"
            st.markdown(
                f'<p style="margin:0;margin-top:-4px;line-height:1.15">{html.escape(str(nm))}</p>',
                unsafe_allow_html=True,
            )
        with c2:
            nm = row.get("home_name") or "—"
            st.markdown(
                f'<p style="margin:0;margin-top:-4px;line-height:1.15">{html.escape(str(nm))}</p>',
                unsafe_allow_html=True,
            )
        with c3:
            st.caption(_format_game_start_pt(row.get("gameDate")))
        with c4:
            st.caption(str(row.get("status") or "—"))
        with c5:
            st.caption(score_line_compact(row))
        with c6:
            st.caption(actual_winner_name(row))
        with c7:
            st.markdown(_format_pick_cell(row, pred_pick, pk), unsafe_allow_html=True)

    st.divider()
    run_sel = st.button(
        "Run projections for selected",
        type="primary",
        key="run_projections_btn",
    )

    game_pks = [int(x) for x in games["gamePk"].tolist()]
    selected = [pk for pk in game_pks if st.session_state.get(f"row_{pk}", False)]

    if run_sel and not selected:
        st.warning("Select at least one game (use checkboxes or Select all).")

    if run_sel and selected:
        sub = games[games["gamePk"].isin(selected)].copy()
        enriched = None
        with st.spinner("Loading MLB stats and scoring selected games…"):
            try:
                enriched = enrich_games_with_features(sub, season, picked)
            except Exception as e:
                st.error(f"Feature build failed: {e}")

        if enriched is not None:
            reg, win_b, runs_b = get_pipelines()
            results: list[dict] = []
            log_rows: list[dict] = []

            for _, row in enriched.iterrows():
                pk = int(row["gamePk"])
                resolved = resolve_game_win(row, reg, win_b)
                proj_h = resolved["proj_h"]
                wp_h = resolved["wp_h"]
                ml_w = resolved["ml_w"]
                home_p = resolved["home_p"]
                away_p = resolved["away_p"]
                pick_name = resolved["pick_name"]
                win_ver = resolved["win_ver"]
                ml_rh, ml_ra, runs_ver = predict_runs_ml(row, runs_b, reg)

                if ml_rh is not None and ml_ra is not None:
                    ph, pa = ml_rh, ml_ra
                else:
                    ph, pa = proj_h["home_exp_runs"], proj_h["away_exp_runs"]
                st.session_state.setdefault("pred_pick", {})
                st.session_state["pred_pick"][pk] = pick_name

                ar, hr = row.get("away_runs"), row.get("home_runs")
                if is_final_row(row) and ar is not None and hr is not None and not (
                    isinstance(ar, float) and np.isnan(ar)
                ) and not (isinstance(hr, float) and np.isnan(hr)):
                    act_away_runs = int(ar)
                    act_home_runs = int(hr)
                    act_winner = actual_winner_name(row)
                else:
                    act_away_runs = "—"
                    act_home_runs = "—"
                    act_winner = "—"

                results.append(
                    {
                        "gamePk": pk,
                        "Away": row.get("away_name"),
                        "Home": row.get("home_name"),
                        "Away probable": row.get("away_probable_name") or "—",
                        "Home probable": row.get("home_probable_name") or "—",
                        "Proj away": round(pa, 2),
                        "Proj home": round(ph, 2),
                        "Total (exp)": round(ph + pa, 2),
                        "Home win %": round(100 * home_p, 1),
                        "Away win %": round(100 * away_p, 1),
                        # Unrounded; used by bet sizing / Kelly (rounded % can become exactly 1.0).
                        "pred_home_win_prob": float(home_p),
                        "pred_away_win_prob": float(away_p),
                        "Pick": pick_name,
                        "Actual winner": act_winner,
                        "Actual away runs": act_away_runs,
                        "Actual home runs": act_home_runs,
                    }
                )

                feats = enriched_row_to_feature_vector(row)
                log = {
                    "logged_at": utc_now_iso(),
                    "gamePk": row.get("gamePk"),
                    "game_date": picked.isoformat(),
                    "season": season,
                    "home_id": row.get("home_id"),
                    "away_id": row.get("away_id"),
                    "home_name": row.get("home_name"),
                    "away_name": row.get("away_name"),
                    "pred_home_win_prob": home_p,
                    "pred_away_win_prob": away_p,
                    "pred_home_win_prob_heur": wp_h["home_win_prob"],
                    "pred_home_win_prob_ml": float(ml_w) if ml_w is not None else np.nan,
                    "pred_home_runs": ph,
                    "pred_away_runs": pa,
                    "pred_home_runs_heur": proj_h["home_exp_runs"],
                    "pred_away_runs_heur": proj_h["away_exp_runs"],
                    "model_win_version": win_ver,
                    "model_runs_version": runs_ver,
                }
                for k, v in feats.items():
                    log[k] = v
                log_rows.append(log)

            st.session_state["proj_results"] = pd.DataFrame(results)
            try:
                log_df = pd.DataFrame(log_rows)
                for c in GAME_FEATURE_NAMES:
                    if c not in log_df.columns:
                        log_df[c] = np.nan
                append_predictions_df(log_df)
                st.session_state["proj_log_note"] = f"Logged {len(log_df)} games → predictions CSV"
                if st.session_state.get("auto_pipeline_bg", True):
                    start_auto_pipeline_background()
                    st.session_state["proj_log_note"] += " · Background merge/train/eval started."
            except Exception as e:
                st.session_state["proj_log_note"] = f"Logging skipped: {e}"

            all_hitters: list[pd.DataFrame] = []
            all_pitchers: list[pd.DataFrame] = []
            prop_errors: list[str] = []
            for _, row in enriched.iterrows():
                hid, aid = row.get("home_id"), row.get("away_id")
                if pd.isna(hid) or pd.isna(aid):
                    prop_errors.append(f"{row.get('away_name')} @ {row.get('home_name')}: missing team ids")
                    continue
                ha = row.get("home_fg") or "NYY"
                pr = project_game_runs(row)
                ml_rh_row, ml_ra_row, _ = predict_runs_ml(row, runs_b, reg)
                if ml_rh_row is not None and ml_ra_row is not None:
                    ph2, pa2 = ml_rh_row, ml_ra_row
                else:
                    ph2, pa2 = pr["home_exp_runs"], pr["away_exp_runs"]
                gpk = int(row["gamePk"])
                pitch_df = pitching_lines_game(gpk)
                try:
                    pdf = props_for_game(
                        season=int(season),
                        home_team_id=int(hid),
                        away_team_id=int(aid),
                        home_team_name=str(row.get("home_name") or "—"),
                        away_team_name=str(row.get("away_name") or "—"),
                        home_fg=str(ha),
                        home_pitcher_id=row.get("home_probable_id"),
                        away_pitcher_id=row.get("away_probable_id"),
                        venue_id=int(row["venue_id"]) if pd.notna(row.get("venue_id")) else None,
                        home_exp_runs=float(ph2),
                        away_exp_runs=float(pa2),
                        top_n=4,
                    )
                except Exception as e:
                    prop_errors.append(f"{row.get('away_name')} @ {row.get('home_name')}: {e!r}")
                    continue
                if not pdf.empty:
                    hit_df = _decorate_hitter_props_for_display(pdf, gpk)
                    all_hitters.append(hit_df)
                prow: list[dict] = []
                for side, pid, pname, opp_x, club in (
                    ("Away", row.get("away_probable_id"), row.get("away_probable_name"), float(ph2), str(row.get("away_name") or "—")),
                    ("Home", row.get("home_probable_id"), row.get("home_probable_name"), float(pa2), str(row.get("home_name") or "—")),
                ):
                    act = pitching_line_for_player(pitch_df, _safe_pid(pid))
                    prow.append(
                        pitcher_prop_display_row(
                            club,
                            side,
                            str(pname) if pname else None,
                            _safe_pid(pid),
                            season,
                            opp_x,
                            act or None,
                        )
                    )
                all_pitchers.append(pd.DataFrame(prow))

            if all_hitters:
                st.session_state["proj_hitters_props"] = pd.concat(all_hitters, ignore_index=True)
            else:
                st.session_state["proj_hitters_props"] = pd.DataFrame()
            if all_pitchers:
                st.session_state["proj_pitchers_props"] = pd.concat(all_pitchers, ignore_index=True)
            else:
                st.session_state["proj_pitchers_props"] = pd.DataFrame()
            st.session_state["proj_props_concat"] = st.session_state["proj_hitters_props"]
            st.session_state["proj_prop_errors"] = prop_errors
            st.session_state["has_proj"] = True

    if st.session_state.get("has_proj") and st.session_state.get("proj_results") is not None:
        st.subheader("Game projections (selected)")
        _show_table(
            st.session_state["proj_results"],
            drop_cols=("gamePk", "pred_home_win_prob", "pred_away_win_prob"),
        )
        if st.session_state.get("proj_log_note"):
            st.caption(st.session_state["proj_log_note"])

        _bet_sizing_section(st.session_state["proj_results"], picked)

        st.subheader("Player props — hitters (selected games)")
        hit_df = st.session_state.get("proj_hitters_props")
        if hit_df is not None and not hit_df.empty:
            _show_table(hit_df)
        else:
            st.caption("No hitter prop rows for this selection.")

        st.subheader("Player props — probable pitchers (selected games)")
        pit_df = st.session_state.get("proj_pitchers_props")
        if pit_df is not None and not pit_df.empty:
            _show_table(pit_df)
        else:
            st.caption("No pitcher rows for this selection.")

        errs = st.session_state.get("proj_prop_errors") or []
        if errs:
            with st.expander("Props errors"):
                st.code("\n".join(errs[:12]), language="text")
    elif not run_sel:
        st.info("Tick games, then **Run projections for selected** for runs, win %, props, and logging.")


if __name__ == "__main__":
    main()
