"""
Fire-and-forget wrapper so Streamlit can kick off merge/train/eval without blocking the UI.
"""

from __future__ import annotations

import threading
import traceback


def start_auto_pipeline_background() -> None:
    """Runs ml.auto_pipeline in a daemon thread (safe for personal use; logs to data/processed/)."""

    def _work() -> None:
        try:
            from ml.auto_pipeline import append_pipeline_log, run_pipeline

            append_pipeline_log(run_pipeline())
        except Exception:
            # Last-resort trace so a silent failure is still diagnosable from the log file.
            from pathlib import Path

            p = Path(__file__).resolve().parents[1] / "data" / "processed" / "auto_pipeline.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write("\n--- THREAD EXCEPTION ---\n")
                f.write(traceback.format_exc())

    threading.Thread(target=_work, daemon=True).start()
