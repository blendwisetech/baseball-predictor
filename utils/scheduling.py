"""
Lightweight helpers for when you run batch jobs (Task Scheduler / cron on WSL).

No scheduler is started from Python — document intended cadence here for Daniel.
"""

# Suggested personal-use cadence (edit to taste):
# - After each slate: ingest_outcomes_for_date (yesterday) then merge_training_data
# - Weekly: python -m ml.train_win_model && python -m ml.train_runs_model
# - Monthly: python -m ml.evaluate_models


def suggested_windows_text() -> str:
    return (
        "**Automation (in the app)**\n"
        "- Enable **Auto merge / train / eval** in the sidebar: after you log projections (or ingest outcomes), "
        "a background job runs `merge → train (if enough rows) → evaluate` and appends to `data/processed/auto_pipeline.log`.\n"
        "\n"
        "**Optional Windows Task Scheduler (nightly)**\n"
        "- Program: `python`  Arguments: `-m ml.auto_pipeline`  "
        "Start in: `C:\\\\Users\\\\dan\\\\.cursor\\\\projects\\\\empty-window\\\\baseball_predictor` (adjust path).\n"
    )
