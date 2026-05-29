"""
Run merge → train win → train runs → evaluate in one process.

Use when you have predictions + outcomes and want production models updated without clicking four buttons.

Run: python -m ml.bootstrap_models
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULES = (
    "ml.merge_training_data",
    "ml.train_win_model",
    "ml.train_runs_model",
    "ml.evaluate_models",
)


def main() -> None:
    for mod in MODULES:
        print(f"\n=== {mod} ===\n", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", mod],
            cwd=str(ROOT),
            timeout=900,
        )
        if r.returncode != 0:
            print(f"Stopped: {mod} exited {r.returncode}", flush=True)
            sys.exit(r.returncode)
    print("\nAll steps completed.", flush=True)


if __name__ == "__main__":
    main()
