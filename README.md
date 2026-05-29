# Baseball predictor

Personal MLB tooling: schedule + MLB Stats API features, Streamlit UI, win/runs models, prediction logging, and training/eval scripts.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run

From the **repository root** (the folder that contains `app/`, `ml/`, and this `README.md`):

```bash
streamlit run streamlit_app.py
```

**Streamlit Cloud:** set **Main file path** to `streamlit_app.py` (not `app/main.py`), so imports resolve from the repo root. If you only see **“Oh no”**, open **Manage app → Logs**; missing **`tzdata`** on Linux breaks `ZoneInfo` at import — `requirements.txt` includes **`tzdata`**. Redeploy after `git pull`.

You can also use `streamlit run app/main.py` **only after** `cd` into the repo root; running it from your home directory without `cd` will fail with “file does not exist”.

Other commands (from repo root, venv activated):

- **Merge training data:** `python -m ml.merge_training_data`
- **Train win model:** `python -m ml.train_win_model`
- **Evaluate:** `python -m ml.evaluate_models`

## Data

CSV/Parquet under `data/` are local logs and processed tables; add your own or run backfill/merge pipelines. Do not commit `.streamlit/secrets.toml` or `.env`.

## License

Repository is public; add a `LICENSE` file if you want explicit terms.
