# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Retrospective XAI demand forecasting on the M5 (Walmart) dataset.

Core question: **"Leader sees the model performed badly at week X — why?"**

## Pipeline (run in order)

```bash
uv run python migrate.py     # create/upgrade DB schema (safe to re-run)
uv run python ingest.py      # download M5, write raw tables (once)
uv run python smoke_test.py  # verify one full cycle before full run
uv run python backtest.py    # full backtest (~125 weeks, ~31 retrains)
uv run streamlit run app.py  # dashboard at localhost:8501
```

## Architecture

```
migrate.py        Applies migrations/*.sql in order, tracks in schema_migrations
ingest.py         M5 download → weekly_sales, calendar, prices, item_meta (raw, no features)
backtest.py       At each iteration: SQL → compute_features() → train → forecast → evaluate → xai
smoke_test.py     Single-cycle sanity check with per-step timing
app.py            Streamlit dashboard

xai_forecast/
  features.py     FEATURE_COLS, constants, compute_features(raw_df) — called at runtime
  db.py           SQLite helpers — get_conn, load_raw_window, insert_*, week_summary
  train.py        train_model(df) → LGBMRegressor
  forecast.py     make_forecasts(model, week_df, week) → [unique_id, h1]
  evaluate.py     evaluate_h1, flag_bad_weeks (rolling z-score)
  xai.py          shap_payloads, counterfactual_payloads, contrastive_payloads

migrations/
  001_raw_tables.sql    weekly_sales, calendar, prices, item_meta (+ indexes)
  002_output_tables.sql forecasts, evaluations, xai_results (+ indexes)

data/             M5 raw files (gitignored — downloaded by ingest.py)
db/               SQLite database (gitignored)
```

## SQLite tables

| Table | Written by | Purpose |
|---|---|---|
| `weekly_sales` | ingest.py | Raw weekly unit sales per SKU |
| `calendar` | ingest.py | SNAP, event flags per week |
| `prices` | ingest.py | Weekly avg sell price per SKU |
| `item_meta` | ingest.py | dept_enc, cat_enc per SKU (static) |
| `forecasts` | backtest.py | h=1 predictions per SKU per cutoff week |
| `evaluations` | backtest.py | MAPE, MAE, bad-week flag per SKU per week |
| `xai_results` | backtest.py | JSON payloads: shap / counterfactual / contrastive |

## Key design decisions

**Store:** CA_1 only (~3,049 SKUs). One global LightGBM model across all SKUs.

**Training window:** Fixed 3-year (156-week) sliding window. Retrain every 4 weeks (`RETRAIN_FREQ`).

**Leakage controls (in build_features.py):**
- Lag features: `shift(n)` — lag_1 at week t = sales[t-1]
- Rolling features: `shift(1).rolling(w)` — excludes current week
- `sell_price` NaN: `ffill` within item (last known price, not global median)
- `price_change_pct`: computed after ffill, `fill_method=None`

**Bad week detection:** Week flagged when avg-MAPE z-score (8-week rolling window) ≥ 1.5.

**XAI (top 50 worst SKUs per bad week):**
- `shap`: TreeSHAP waterfall — what drove the prediction
- `counterfactual`: zero out SNAP/event/price-change → measure prediction delta
- `contrastive`: compare SHAP profile vs a similar good week (same week-of-year, MAPE < 15%)

**Adding a migration:** Add `00N_description.sql` to `migrations/`, run `migrate.py`. Never edit applied migrations.

## Stack

- Python + `uv` (never pip/venv)
- LightGBM + SHAP
- SQLite (stdlib) — WAL mode enabled
- Streamlit + Plotly
