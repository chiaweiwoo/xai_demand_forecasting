# XAI Demand Forecasting

Retrospective explainability on M5 (Walmart) weekly demand. Answers the question a business leader actually asks: **"The model performed badly at week X — why?"**

## What it does

Runs a full sliding-window backtest over 5 years of Walmart CA_1 store sales (~3,049 SKUs, ~120 weeks). For every week where the model's WMAPE spikes anomalously, it produces three types of explanation for the top-50 worst SKUs:

- **SHAP** — which features drove the prediction up or down
- **Counterfactual** — "if there had been no SNAP/event/price change, how different would the forecast have been?"
- **Contrastive** — "compared to a similar week where the model got it right, what was different?"

Results are stored in SQLite and explored through a Streamlit dashboard.

## Setup

```bash
# Install dependencies
uv sync

# Download M5 data and ingest into SQLite (run once)
uv run python ingest.py

# Precompute feature store (run once; re-run whenever features.py changes)
uv run python build_features.py

# Sanity check (10 weeks in parallel, ~25s)
uv run python smoke_test.py

# Full backtest (~120 weeks, ~30 retrains)
uv run python backtest.py

# Launch dashboard
uv run streamlit run app.py
```

## Pipeline

```
ingest.py          M5 CSVs → SQLite (weekly_sales, calendar, prices, item_meta)
build_features.py  Precompute all features once → features table (847k rows, ~46s)
backtest.py        Sliding-window train/forecast/evaluate/explain → output tables
smoke_test.py      10-week parallel sanity check (writes to isolated smoke.db)
app.py             Streamlit dashboard
```

## Model

- **Algorithm:** LightGBM, Tweedie objective (variance_power=1.5) — correct for 64% zero-sale intermittent data
- **Scope:** One global model across all SKUs; CA_1 store only
- **Training window:** 3-year (156-week) sliding, retrained every 4 weeks
- **Features (19):** lag_1/2/4/8/52, rolling means/std (4/8/13 weeks), week-of-year, month, year, SNAP, event flags, sell price, price change %, dept/cat mean sales
- **Bad week flag:** WMAPE z-score ≥ 1.5 on an 8-week rolling window

## Stack

- Python 3.11+ with `uv`
- LightGBM + SHAP
- SQLite (WAL mode)
- Streamlit + Plotly

## Data

M5 Forecasting Competition dataset (Walmart sales 2011–2016). Downloaded automatically by `ingest.py` from Kaggle. Raw files and the SQLite database are gitignored.
