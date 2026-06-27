# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Retrospective XAI demand forecasting on the M5 (Walmart) dataset.

Core question: **"Leader sees the model performed badly at week X — why?"**

## Pipeline (run in order)

```bash
uv run python ingest.py          # download M5, write raw tables (once only)
uv run python build_features.py  # precompute all features → features table (rebuild whenever features.py changes)
uv run python smoke_test.py      # sanity check before full run (staleness, contrastive, SHAP additivity)
uv run python backtest.py        # full backtest (~120 weeks, ~30 retrains)
uv run python data_quality.py    # post-backtest integrity checks (run before opening dashboard)
uv run streamlit run app.py      # dashboard at localhost:8501
uv run pytest                    # 73 tests covering features, evaluation, XAI contracts, DB, end-to-end, narratives
```

**Critical invariant: rebuild the feature store whenever `features.py` changes.**
`build_features.py` always clears and rebuilds — safe to re-run at any time. Forgetting this means backtest trains on stale/incorrect features silently. The smoke test catches this via a live diff before committing to a full run.

## Architecture

```
ingest.py          M5 CSVs → weekly_sales, calendar, prices, item_meta (raw, no features)
build_features.py  One-time: compute_features() on all 847k rows → features table
backtest.py        Sliding-window train/forecast/evaluate/xai → output tables
                   Keeps all retrain checkpoints in memory; XAI uses the exact model
                   that produced each week's forecast (per-checkpoint, not one final model).
smoke_test.py      Sanity check: feature staleness, parallel forecast, contrastive, SHAP additivity,
                   + narrative API probe (one live DeepSeek call — fails loudly if config is wrong)
data_quality.py    Post-backtest: referential integrity, h1>=0, pre-launch price leakage, etc.
app.py             Streamlit dashboard (4 pages — see Dashboard section)

xai_forecast/
  features.py      FEATURE_COLS, compute_features(raw_df) — single source of truth for all features
  db.py            SQLite helpers: get_conn (auto-applies schema), load_features_window,
                   load_features_week, insert_*, week_summary, load_all_shap_payloads,
                   insert_narrative, load_narrative, load_narratives_by_scope
  train.py         train_model(df) → LGBMRegressor (Tweedie objective)
  forecast.py      make_forecasts(model, week_df, week) → [unique_id, h1]
  evaluate.py      evaluate_h1, flag_bad_weeks (rolling WMAPE z-score, prior-weeks-only baseline)
  xai.py           shap_payloads → (rows, shap_cache), counterfactual_payloads, contrastive_payloads
                   Payloads include signed_error + direction (over/under) for narrative layer.
  narrate.py       LLM narrative layer (DeepSeek V4 Flash). Build dossiers (pure) → generate narratives.
                   Graceful no-key fallback — returns None if DEEPSEEK_API_KEY is unset.
                   WEEK_NARRATIVE_PROMPT, ITEM_NARRATIVE_PROMPT, EXECUTIVE_NARRATIVE_PROMPT constants.

tests/
  conftest.py           Shared fixtures: raw_df, trained_model_and_explainer, db_conn
  test_features.py      Group A: lag correctness, rolling leakage, bfill regression, future-invariance
  test_evaluate.py      Group B: evaluate_h1, WMAPE formula, z-score, NaN propagation
  test_xai_payloads.py  Group C: SHAP/CF payload contract, additivity, json round-trip
  test_db.py            Group D: INSERT OR REPLACE, read-back, clean-slate DELETE, features shape
  test_contrastive.py   Contrastive: WOY selection, skip-when-no-match, shap_diff math, cache equality
  test_correctness.py   Regression: baseline excludes current week, NaN forecast, wiring, end-to-end
  test_narrate.py       Narrate: dossier builders (pure), grounding check, no-key fallback, mock LLM

migrations/
  001_raw_tables.sql     weekly_sales, calendar, prices, item_meta (+ indexes)
  002_output_tables.sql  forecasts, evaluations, xai_results (+ indexes)
  003_features_table.sql features (+ index on week)
  004_narratives.sql     narratives (scope, key, payload, model, created_at) — PRIMARY KEY (scope, key)

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
| `narratives` | backtest.py | LLM-generated narratives (scope: week/item/executive), keyed by (scope, key) |

## Key design decisions

**Store:** CA_1 only (~3,049 SKUs). One global LightGBM model across all SKUs.

**Training window:** Fixed 3-year (156-week) sliding window. Retrain every 4 weeks (`RETRAIN_FREQ`). 278 total weeks → ~120 backtest weeks, ~30 retrains.

**Week ID:** Saturday date string — Walmart fiscal week start, derived from `wm_yr_wk` in the calendar. e.g. `2011-01-29`. All tables join on this string.

**Partial last week excluded:** The M5 evaluation file ends 2 days into the final fiscal week (2016-05-21). Forecasting it produces a spurious ~215% MAPE spike. `backtest_weeks = weeks[TRAIN_WINDOW:-2]` — the last two weeks are excluded as backtest targets.

**Feature store:** `build_features.py` precomputes all 847k feature rows once (~46s). `backtest.py` and `smoke_test.py` do a plain SQL SELECT per iteration instead of recomputing features. Per-iteration time went from ~36s to ~2s. **Must be rebuilt whenever `features.py` changes** — `build_features.py` always clears and rebuilds, so just re-run it. The smoke test catches staleness before a full run.

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

**Objective:** Tweedie (variance_power=1.5) — correct for zero-heavy intermittent count data (64% zero-sale days). SHAP values are in **log-margin space** (Tweedie log-link). `base_value_log + Σ(shap_values) = log(prediction)`. Feature ranking by `|shap|` is valid.

**Bad week detection:** Week flagged when WMAPE z-score ≥ 1.5 on an 8-week rolling window (min 3 periods). Baseline uses `shift(1)` before rolling so the current week is **excluded from its own baseline** — a spike is scored against the prior period, not inflated against itself. WMAPE = Σ|error| / Σactual — volume-weighted, not dominated by near-zero-actual SKUs the way avg-MAPE is.

**Week key convention:** All output tables (forecasts, evaluations, xai_results) are keyed on `forecast_week` — the week the error was observed. Not the training cutoff. This is the natural "week X" a leader would point at.

**Idempotency:** `backtest.py` deletes all rows from forecasts/evaluations/xai_results/narratives at the start of each run. Re-running always produces a clean result with no orphan rows.

**Smoke test isolation:** `smoke_test.py` reads from `db/forecasting.db` (source DB, never written to) and writes all output to `db/smoke.db` (throwaway). Running the smoke test never contaminates dashboard data.

**XAI model — per retrain checkpoint:** SHAP/counterfactual/contrastive for each bad week are computed using the exact retrain checkpoint that produced that week's forecast (at 4-week granularity). All ~30 checkpoints are kept in memory during the backtest run. Explainers are cached by checkpoint to avoid rebuilding per bad week. Contrastive compares both the bad week and its reference week under the **same model** (the bad week's checkpoint) to keep SHAP profiles in the same space.

**XAI (top 50 worst SKUs per bad week):**
- `shap`: TreeSHAP — top 5 drivers in log-margin space, plus `other_features_shap` (sum of remaining 14) so the waterfall reconciles: `base_value_log + Σ(top5) + other_features_shap ≈ log(prediction)`
- `counterfactual`: zero out SNAP / event / price-change → measure prediction delta. Each scenario includes `was_active: bool` so the dashboard can distinguish meaningful zeroing from a no-op.
- `contrastive`: compare SHAP profile vs a good reference week for the same SKU (same ISO week-of-year, MAPE < 15% in full eval history). Skips items with no same-WOY good week — no fallback to different-seasonality weeks. `contrastive_payloads` must receive `all_evals_df` (full history), not just the current bad week's evals — otherwise good reference weeks are never found. `seasonality_matched: True` in every payload (guaranteed by the skip logic).

**shap_payloads API:** Returns `(list[dict], dict[str, np.ndarray])` — the DB rows and a `shap_cache` mapping uid → raw SHAP array. Pass `shap_cache` to `contrastive_payloads` as `bad_shap_cache` to avoid recomputing bad-item SHAP. Both `backtest.py` and `smoke_test.py` do this.

**SHAP payload extras:** Each SHAP row now includes `signed_error` (signed % error, positive = over-forecast) and `direction` ("over"/"under") so the LLM narrative layer can reference error direction without computing it.

**LLM narrative layer (`xai_forecast/narrate.py`):**
- `DeepSeekNarrator` wraps DeepSeek V4 Flash (`deepseek-v4-flash`) via the OpenAI SDK (base_url = `https://api.deepseek.com`). Key via `DEEPSEEK_API_KEY` env var. Set in `.env` (copy from `.env.example`).
- Three prompt constants: `WEEK_NARRATIVE_PROMPT`, `ITEM_NARRATIVE_PROMPT`, `EXECUTIVE_NARRATIVE_PROMPT`. These are `*_PROMPT` constants — run `/prompt-audit` before editing them.
- Three dossier builders (pure functions, no network): `build_week_dossier`, `build_item_dossier`, `build_executive_dossier`.
- Post-generation grounding check: verifies `primary_driver` is in the evidence feature list. Sets `confidence=low` + `grounding_warning=True` if it fails.
- Narratives are generated during `backtest.py` (after XAI phase) and cached in the `narratives` table.
- If `DEEPSEEK_API_KEY` is not set, narratives are skipped silently. Dashboard falls back to charts only.
- Smoke test includes one live API probe (real call, fails loudly on bad config).
- `compute_recurring_drivers(shap_rows)`: single source of truth for recurring-driver aggregation. Returns `{feature, count, pct_payloads, n_weeks, pct_bad_weeks}` per feature. `pct_bad_weeks` = % of distinct bad weeks in which this feature appeared as a top-5 driver — used for executive narrative confidence thresholding (high >60%, medium 40–60%, low otherwise).

**Pre-launch SKUs:** A SKU in its pre-launch weeks has all-NaN lag features. `make_forecasts` imputes these to 0 (via `.fillna(0)` before `model.predict`). If such a SKU has `y > 0` in that week, it is evaluated against a garbage forecast. `backtest.py` counts and logs these rows. `data_quality.py` checks for pre-launch price leakage (sell_price non-null with lag_1 null — would indicate the bfill bug returning).

## Dashboard pages

| Page | What it shows |
|---|---|
| Overview | Weekly MAPE time series with bad-week markers |
| Bad Week Drilldown | LLM week narrative (headline card) + worst items + MAPE distribution + week-level SHAP aggregation |
| Recurring Drivers | LLM executive synthesis + feature appearance frequency across all bad weeks |
| XAI Explorer | LLM item narrative + per-item SHAP waterfall, counterfactual (inactive grayed), contrastive |

## Stack

- Python + `uv` (never pip/venv)
- LightGBM + SHAP
- SQLite (stdlib) — WAL mode enabled
- Streamlit + Plotly
- DeepSeek V4 Flash via `openai` SDK (narrative layer, optional)
