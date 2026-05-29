"""
Player props are multi-entity (hitter × game). This module documents the path and
trains a minimal placeholder when per-player labels exist in processed data.

For a full props model, log one row per (gamePk, player_id, prop_type) with features
and later join Statcast / boxscore lines — extend feature_config when ready.

Run: python -m ml.train_props_model
"""

from __future__ import annotations

from pathlib import Path

from utils.data_io import ROOT, ensure_dirs, load_registry, save_registry


def main() -> None:
    ensure_dirs()
    props_path = ROOT / "data" / "processed" / "training_props.parquet"
    if not props_path.exists():
        print(
            "No training_props.parquet yet. "
            "When you log per-player prop rows with labels, save them here, then re-run."
        )
        reg = load_registry()
        reg.setdefault("production", {})
        reg["production"]["props_model_version"] = "heuristic_v1"
        reg["production"]["props_model_path"] = None
        save_registry(reg)
        return
    print("Props training stub: implement after labeled prop rows exist.")


if __name__ == "__main__":
    main()
