# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Retrospective XAI demand forecasting system using the M5 (Walmart) dataset.

The core question it answers: **"Leader sees the model performed badly at week X — why?"**

Weekly backtest loop: train LightGBM on expanding history → forecast 3 weeks → compare actual vs predicted → flag bad weeks (MAPE z-score ≥ 1.5) → run XAI on the worst 50 items per bad week → write everything to SQLite → display in Streamlit.

## Commands

```bash
# Install deps
uv sync

# Download M5 data and run full backtest (~10–20 min)
uv run python backtest.py

# Launch dashboard
uv run streamlit run app.py
```

## Architecture

```
xai_forecast/       Python package
  features.py       M5 data loading, weekly aggregation, feature engineering
  db.py             SQLite helpers (init_db, insert_*, load_*, week_summary)
  train.py          LightGBM wrapper (train_model)
  forecast.py       make_forecasts — vectorised, returns [unique_id, h1, h2, h3]
  evaluate.py       evaluate_h1, flag_bad_weeks (rolling z-score)
  xai.py            shap_payloads, counterfactual_payloads, contrastive_payloads

backtest.py         Orchestrator: expanding-window loop → writes db/forecasting.db
app.py              Streamlit dashboard (Overview / Bad Week Drilldown / XAI Explorer)

data/               M5 raw files — downloaded by datasetsforecast (gitignored)
db/                 SQLite database (gitignored)
```

## Key design decisions

**Data**: M5 store `CA_1`, ~3k SKUs, 5.5 years daily aggregated to weekly. Loaded via `datasetsforecast` (no Kaggle login needed).

**Features** (19 total): lag_{1,2,4,8,52}, rolling_{4,8,13}_mean, rolling_4_std, week_of_year, month, year, snap, has_event, event_type_enc, sell_price, price_change_pct, dept_enc, cat_enc. All defined in `FEATURE_COLS` in `features.py`.

**Backtest**: Retrains every 4 weeks (`RETRAIN_FREQ`), 52-week warmup. Uses actual lags for h=2/h=3 (retrospective assumption — valid for diagnosis purposes, would be leaky in production).

**Bad week detection**: A week is flagged when its avg-MAPE z-score (8-week rolling window) ≥ 1.5. Threshold in `evaluate.py:flag_bad_weeks`.

**XAI — three angles per bad week:**
- `shap`: TreeSHAP waterfall for the h=1 prediction (top 5 features, stored as JSON)
- `counterfactual`: perturb SNAP/event/price_change to zero; measure prediction delta — answers "how much did this feature inflate the forecast?"
- `contrastive`: find a historical good week (same week-of-year, MAPE < 15%) for the same item; diff SHAP profiles — answers "what was structurally different that time?"

**SQLite schema**: `forecasts`, `actuals`, `evaluations`, `xai_results` — all keyed by `(week_id, item_id)`. XAI payloads stored as JSON strings in `xai_results.payload`.

## Stack

- Python managed with `uv` (never `pip` or `venv`)
- LightGBM + SHAP for model and explanations
- SQLite (stdlib) for persistence
- Streamlit + Plotly for the dashboard
