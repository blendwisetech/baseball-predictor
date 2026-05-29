"""
Incremental / streaming updates.

sklearn GradientBoosting* does **not** support partial_fit. For continuous learning with GBM,
re-run batch trainers (`python -m ml.train_win_model`) on an updated merged parquet.

If you switch production to **SGDClassifier** or **PassiveAggressiveClassifier**, you can call
`partial_fit` on rolling windows — add that training path here when you promote those models.
"""

from __future__ import annotations


def note() -> str:
    return (
        "Online partial_fit is optional; batch retrain on merged data is the default path "
        "for tree ensembles used in train_win_model / train_runs_model."
    )


if __name__ == "__main__":
    print(note())
