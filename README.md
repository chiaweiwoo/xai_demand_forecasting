# XAI Demand Forecasting

Retrospective explainability on M5 (Walmart) weekly demand. Answers the question a business leader actually asks: **"The model performed badly at week X — why?"**

## What it does

Runs a full sliding-window backtest over 5 years of Walmart CA_1 store sales (~3,049 SKUs, ~120 weeks). For every week where the model's WMAPE spikes anomalously, it produces three types of explanation for the top-50 worst SKUs:

- **SHAP** — which features drove the prediction up or down (with full waterfall reconciliation to the prediction)
- **Counterfactual** — "if there had been no SNAP/event/price change, how different would the forecast have been?"
- **Contrastive** — "compared to a similar week where the model got it right, what was structurally different?"

XAI explanations come from the actual retrain checkpoint that produced each week's forecast — not a single final model — so the explanations are faithful to what the model knew at the time.

Results are stored in SQLite and explored through a four-page Streamlit dashboard. Each bad week gets an LLM-generated plain-English narrative (DeepSeek V4 Flash) explaining the failure in business terms — no jargon, grounded only in the XAI evidence.

## Setup

```bash
# Install dependencies
uv sync

# (Optional) Configure LLM narrative layer
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY — narratives are skipped if key is absent

# Download M5 data and ingest into SQLite (run once)
uv run python ingest.py

# Precompute feature store (run once; re-run whenever features.py changes)
uv run python build_features.py

# Sanity check before full run (includes live API probe if DEEPSEEK_API_KEY is set)
uv run python smoke_test.py

# Full backtest (~120 weeks, ~30 retrains) + narrative generation
uv run python backtest.py

# Post-backtest data quality checks
uv run python data_quality.py

# Launch dashboard
uv run streamlit run app.py
```

## Pipeline

```
ingest.py          M5 CSVs → SQLite (weekly_sales, calendar, prices, item_meta)
build_features.py  Precompute all features once → features table (847k rows, ~46s)
smoke_test.py      Sanity check: feature staleness, parallel forecast, contrastive, SHAP additivity
backtest.py        Sliding-window train/forecast/evaluate/explain → output tables
data_quality.py    Integrity checks: h1>=0, XAI referential integrity, pre-launch leakage
app.py             Streamlit dashboard (4 pages)
```

## Model

- **Algorithm:** LightGBM, Tweedie objective (variance_power=1.5) — correct for 64% zero-sale intermittent data
- **Scope:** One global model across all SKUs; CA_1 store only
- **Training window:** 3-year (156-week) sliding, retrained every 4 weeks (~30 retrains total)
- **XAI model:** Each bad week explained by its own retrain checkpoint (not one final model)
- **Features (19):** lag_1/2/4/8/52, rolling means/std (4/8/13 weeks), week-of-year, month, year, SNAP, event flags, sell price, price change %, dept/cat mean sales
- **Bad week flag:** WMAPE z-score ≥ 1.5 on a prior-weeks-only 8-week rolling baseline

## Dashboard

| Page | Purpose |
|---|---|
| Overview | Weekly MAPE time series with bad-week markers |
| Bad Week Drilldown | LLM narrative card + worst items table + week-level SHAP aggregation |
| Recurring Drivers | LLM executive synthesis + feature frequency across all bad weeks (% of bad weeks each feature appeared in) |
| XAI Explorer | LLM item narrative + SHAP waterfall, counterfactual, contrastive |

## Testing

```bash
uv run pytest          # ~83 tests, ~4s
```

Test groups:
- **A (features):** lag correctness, rolling leakage, bfill regression, future-invariance
- **B (evaluate):** WMAPE formula, z-score, NaN propagation, drop-zero-actual
- **C (XAI payloads):** all dashboard-read keys (incl. signed_error/direction), SHAP additivity, JSON round-trip
- **D (DB):** INSERT OR REPLACE idempotency, read-back, clean-slate DELETE, narrative CRUD
- **Contrastive:** same-WOY selection, skip-when-no-match, shap_diff math, cache equality
- **Correctness regression:** baseline shift(1) property, NaN forecast handling, end-to-end mini-backtest
- **Narrate:** dossier builders (pure), grounding check, no-key fallback, mocked LLM schema + round-trip, pct_bad_weeks metric

## Stack

- Python 3.11+ with `uv`
- LightGBM + SHAP
- SQLite (WAL mode)
- Streamlit + Plotly
- DeepSeek V4 Flash via `openai` SDK (narrative layer, optional — set `DEEPSEEK_API_KEY`)

## Data

M5 Forecasting Competition dataset (Walmart sales 2011–2016). Downloaded automatically by `ingest.py` from Kaggle. Raw files and the SQLite database are gitignored.
