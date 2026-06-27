# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Retrospective XAI demand forecasting on the M5 (Walmart) dataset.

Core question: **"Leader sees the model performed badly at week X — why?"**

## Pipeline (run in order)

```bash
uv run python ingest.py          # download M5, write raw tables (once only)
uv run python build_features.py  # precompute all features → features table (rebuild whenever features.py changes)
uv run python smoke_test.py      # 10-week parallel sanity check before full run
uv run python backtest.py        # full backtest (~120 weeks, ~30 retrains)
uv run streamlit run app.py      # dashboard at localhost:8501
```

**Critical invariant: rebuild the feature store whenever `features.py` changes.**
`build_features.py` always clears and rebuilds — safe to re-run at any time. Forgetting this means backtest trains on stale/incorrect features silently.

## Architecture

```
ingest.py          M5 CSVs → weekly_sales, calendar, prices, item_meta (raw, no features)
build_features.py  One-time: compute_features() on all 847k rows → features table
backtest.py        At each iteration: load from features table → train → forecast → evaluate → xai
smoke_test.py      10-week parallel sanity check; reads SOURCE_DB, writes to isolated SMOKE_DB
app.py             Streamlit dashboard

xai_forecast/
  features.py      FEATURE_COLS, compute_features(raw_df) — single source of truth for all features
  db.py            SQLite helpers: get_conn (auto-applies schema), load_features_window,
                   load_features_week, insert_*, week_summary
  train.py         train_model(df) → LGBMRegressor (Tweedie objective)
  forecast.py      make_forecasts(model, week_df, week) → [unique_id, h1]
  evaluate.py      evaluate_h1, flag_bad_weeks (rolling WMAPE z-score)
  xai.py           shap_payloads, counterfactual_payloads, contrastive_payloads

migrations/
  001_raw_tables.sql     weekly_sales, calendar, prices, item_meta (+ indexes)
  002_output_tables.sql  forecasts, evaluations, xai_results (+ indexes)
  003_features_table.sql features (+ index on week)

data/   M5 raw files (gitignored — downloaded by ingest.py)
db/     SQLite databases (gitignored): forecasting.db (production), smoke.db (throwaway)
```

Schema is applied automatically by `get_conn()` via `_setup_schema()` — no manual migration step. To add schema changes: add `00N_description.sql` to `migrations/`. Never edit existing migration files.

## SQLite tables

| Table | Written by | Purpose |
|---|---|---|
| `weekly_sales` | ingest.py | Raw weekly unit sales per SKU |
| `calendar` | ingest.py | SNAP, event flags per week |
| `prices` | ingest.py | Weekly avg sell price per SKU |
| `item_meta` | ingest.py | dept_id, cat_id, dept_mean_sales, cat_mean_sales per SKU |
| `features` | build_features.py | Precomputed feature matrix for all 847k (unique_id, week) rows |
| `forecasts` | backtest.py | h=1 predictions per SKU per forecast week |
| `evaluations` | backtest.py | MAPE, MAE, WMAPE z-score, bad-week flag per SKU per week |
| `xai_results` | backtest.py | JSON payloads: shap / counterfactual / contrastive |

## Key design decisions

**Store:** CA_1 only (~3,049 SKUs). One global LightGBM model across all SKUs.

**Training window:** Fixed 3-year (156-week) sliding window. Retrain every 4 weeks (`RETRAIN_FREQ`). 278 total weeks → ~120 backtest weeks, ~30 retrains.

**Week ID:** Saturday date string — Walmart fiscal week start, derived from `wm_yr_wk` in the calendar. e.g. `2011-01-29`. All tables join on this string.

**Partial last week excluded:** The M5 evaluation file ends 2 days into the final fiscal week (2016-05-21). Forecasting it produces a spurious ~215% MAPE spike. `backtest_weeks = weeks[TRAIN_WINDOW:-2]` — the last two weeks are excluded as backtest targets.

**Feature store:** `build_features.py` precomputes all 847k feature rows once (~46s). `backtest.py` and `smoke_test.py` do a plain SQL SELECT per iteration instead of recomputing features. Per-iteration time went from ~36s to ~2s. **Must be rebuilt whenever `features.py` changes** — `build_features.py` always clears and rebuilds, so just re-run it.

**Features (19 total):**
- Lags (5): lag_1, lag_2, lag_4, lag_8, lag_52 — lag_52 is the same-week-last-year seasonality anchor
- Rolling (4): rolling_4/8/13_mean, rolling_4_std — all use `shift(1)` before `.rolling()` to exclude current week
- Calendar (3): week_of_year, month, year
- Store context (3): snap, has_event, event_type_enc
- Price (2): sell_price (ffill within item only — no bfill), price_change_pct
- Item metadata (2): dept_mean_sales, cat_mean_sales

**Leakage controls:**
- Lag features: `shift(n)` per SKU — lag_1 at week t = sales[t-1]
- Rolling features: `shift(1).rolling(w)` — excludes current week
- `sell_price` NaN: `ffill` within item only — no `.bfill()`. Pre-launch NaNs stay NaN and are dropped by `dropna(FEATURE_COLS)` at training time. Using `bfill` here would pull a future price backward into 87k pre-launch rows across 60% of SKUs, allowing them to survive `dropna` and enter training with leaked data.
- `price_change_pct`: computed after ffill, `fill_method=None`
- dept_mean_sales / cat_mean_sales: static prior computed over full history — a mild, deliberate lookahead accepted for stability (only 7 dept / 3 cat scalars; demand scale is stationary). Do not claim zero leakage for these.

**Objective:** Tweedie (variance_power=1.5) — correct for zero-heavy intermittent count data (64% zero-sale days). SHAP values are in **log-margin space** (Tweedie log-link). `base_value_log + Σ(shap_values) = log(prediction)`. Feature ranking by `|shap|` is valid; `base_value_log` and `prediction` are in different units and do not add up directly.

**Bad week detection:** Week flagged when WMAPE z-score (8-week rolling window, min 3 periods) ≥ 1.5. WMAPE = Σ|error| / Σactual — volume-weighted, not dominated by near-zero-actual SKUs the way avg-MAPE is.

**Week key convention:** All output tables (forecasts, evaluations, xai_results) are keyed on `forecast_week` — the week the error was observed. Not the training cutoff. This is the natural "week X" a leader would point at.

**Idempotency:** `backtest.py` deletes all rows from forecasts/evaluations/xai_results at the start of each run. Re-running always produces a clean result with no orphan rows.

**Smoke test isolation:** `smoke_test.py` reads from `db/forecasting.db` (source DB, never written to) and writes all output to `db/smoke.db` (throwaway). Running the smoke test never contaminates dashboard data.

**XAI model caveat:** SHAP/counterfactual/contrastive are computed with a single model retrained on the most recent 3-year window, not the model that originally produced each week's forecast. The dashboard must not claim it is the exact model that erred. Feature-importance relationships are stable enough for the explanation to be useful.

**XAI (top 50 worst SKUs per bad week):**
- `shap`: TreeSHAP — top 5 drivers in log-margin space
- `counterfactual`: zero out SNAP / event / price-change → measure prediction delta
- `contrastive`: compare SHAP profile vs a good reference week for the same SKU (same ISO week-of-year, MAPE < 15% in full eval history). `contrastive_payloads` must receive `all_evals_df` (full history), not just the current bad week's evals — otherwise good reference weeks are never found.

## Stack

- Python + `uv` (never pip/venv)
- LightGBM + SHAP
- SQLite (stdlib) — WAL mode enabled
- Streamlit + Plotly
