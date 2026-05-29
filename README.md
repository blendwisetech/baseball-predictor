# Baseball predictor

Personal MLB tooling: schedule + MLB Stats API features, Streamlit UI, win/runs models, prediction logging, and training/eval scripts.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run

- **Streamlit app:** `streamlit run app/main.py` (or your Cloud entry script if you mirror this repo there).
- **Merge training data:** `python -m ml.merge_training_data`
- **Train win model:** `python -m ml.train_win_model`
- **Evaluate:** `python -m ml.evaluate_models`

## Data

CSV/Parquet under `data/` are local logs and processed tables; add your own or run backfill/merge pipelines. Do not commit `.streamlit/secrets.toml` or `.env`.

## License

Repository is public; add a `LICENSE` file if you want explicit terms.
