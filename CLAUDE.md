# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Retrospective XAI demand forecasting on the M5 (Walmart) dataset.

Core question: **"Leader sees the model performed badly at week X — why?"**

## Pipeline (run in order)

```bash
uv run python ingest.py               # download M5, write raw tables (once only)
uv run python build_features.py       # precompute all features → features table (rebuild whenever features.py changes)
uv run python smoke_test.py           # sanity check before full run (staleness, contrastive, SHAP additivity)
uv run python backtest.py             # forecast + evaluate → forecasts, evaluations; saves models/ + week_to_cutoff.json
uv run python run_xai.py              # SHAP / counterfactual / contrastive → xai_results (re-runnable independently)
uv run python generate_insights.py    # evidence-first insights → insight_findings, insight_summary (re-runnable independently)
uv run python data_quality.py         # post-run integrity checks (run before opening dashboard)
uv run streamlit run app.py           # dashboard at localhost:8501
uv run streamlit run code_review.py --server.port 7501  # code walkthrough app
uv run pytest                         # 84 tests covering features, evaluation, XAI contracts, DB, end-to-end, insights
```

**Each stage is independently re-runnable.** If only insights need fixing, re-run `generate_insights.py` alone — no need to redo ML or XAI. If XAI logic changes, re-run `run_xai.py` + `generate_insights.py`. Only re-run `backtest.py` if model training or evaluation logic changes.

**Critical invariant: rebuild the feature store whenever `features.py` changes.**
`build_features.py` always clears and rebuilds — safe to re-run at any time. Forgetting this means backtest trains on stale/incorrect features silently. The smoke test catches this via a live diff before committing to a full run.

## Architecture

```
ingest.py               M5 CSVs → weekly_sales, calendar, prices, item_meta (raw, no features)
build_features.py       One-time: compute_features() on all 847k rows → features table
backtest.py             Sliding-window train/forecast/evaluate → forecasts, evaluations tables
                        Saves per-retrain LightGBM checkpoints to models/ dir and week_to_cutoff.json.
run_xai.py              Loads saved checkpoints → SHAP / counterfactual / contrastive → xai_results
                        Re-runnable independently of backtest. Uses exact per-checkpoint model per week.
generate_insights.py    Reads xai_results → evidence-first insights → insight_findings, insight_summary.
                        Requires DEEPSEEK_API_KEY. Fails loudly if absent. Re-runnable independently.
                        Logs full agent trace to logs/insights.log (overwritten each run).
smoke_test.py           Sanity check: feature staleness, parallel forecast, contrastive, SHAP additivity,
                        + one live DeepSeek API probe (fails loudly if config is wrong)
data_quality.py         Post-run: referential integrity, h1>=0, pre-launch price leakage, etc.
app.py                  Streamlit dashboard (4 pages — see Dashboard section)

ingest_external.py      Stage 1: reads committed external_data/*.csv → external_signals table.
                        Sat–Fri weather roll-up, fiscal-week gas mapping, monthly sentiment ffill. Idempotent.
                        NOTE: external_signals is NOT yet consumed by features.py (Stage 2 will wire it in).
validate_external.py    Stage 1 gate: 21 PASS/FAIL checks (coverage + structural + real-world anchors).
tools/
  fetch_external_raw.py Run-once internet fetch (Open-Meteo LA weather, EIA CA gas, FRED sentiment) →
                        external_data/ CSVs. NOT part of the pipeline — the pipeline reads the committed CSVs offline.

xai_forecast/
  features.py      FEATURE_COLS, compute_features(raw_df) — single source of truth for all features
  db.py            SQLite helpers: get_conn (auto-applies schema), load_features_window,
                   load_features_week, insert_*, week_summary, load_all_shap_payloads,
                   insert_insight_finding, insert_insight_summary, load_insight_findings, load_insight_summary
  train.py         train_model(df) → LGBMRegressor (Tweedie objective)
  forecast.py      make_forecasts(model, week_df, week) → [unique_id, h1]
  evaluate.py      evaluate_h1, flag_bad_weeks (rolling WMAPE z-score, prior-weeks-only baseline)
  xai.py           shap_payloads → (rows, shap_cache), counterfactual_payloads, contrastive_payloads
                   Payloads include signed_error + direction (over/under) for insights module.
  insights/        Evidence-first insights module (replaces the former narrate.py layer):
    schemas.py       CandidateFinding, Hypothesis, Critique, LedgerRow dataclasses
    detectors.py     Deterministic detectors: over_forecast_bias, dominant_driver, demand_cliff,
                     external_coincidence, counterfactual_material, contrastive_gap
    tools.py         Read-tools called by the planner: read_forecast_accuracy, read_bad_weeks,
                     read_xai_findings, read_demand_trajectory, read_external_signals,
                     read_model_metadata, read_recurring_drivers
    llm_client.py    DeepSeekClient: call_flash (deepseek-v4-flash), call_pro (deepseek-v4-pro)
    agents.py        run_planner, run_hypothesis, run_critic, run_synthesis + all *_PROMPT constants
                     run /prompt-audit before editing any *_PROMPT constant
    graph.py         LangGraph StateGraph: detect_candidates → fan-out review_finding → synthesize
                     Grounding check is advisory only — result forwarded to Pro critic, not a gate.

tests/
  conftest.py           Shared fixtures: raw_df, trained_model_and_explainer, db_conn
  test_features.py      Group A: lag correctness, rolling leakage, bfill regression, future-invariance
  test_evaluate.py      Group B: evaluate_h1, WMAPE formula, z-score, NaN propagation
  test_xai_payloads.py  Group C: SHAP/CF payload contract, additivity, json round-trip
  test_db.py            Group D: INSERT OR REPLACE, read-back, clean-slate DELETE, features shape
  test_contrastive.py   Contrastive: WOY selection, skip-when-no-match, shap_diff math, cache equality
  test_correctness.py   Regression: baseline excludes current week, NaN forecast, wiring, end-to-end
  test_insights.py      Insights: detector contracts, tool read functions, DB round-trip, graph smoke

migrations/
  001_raw_tables.sql          weekly_sales, calendar, prices, item_meta (+ indexes)
  002_output_tables.sql       forecasts, evaluations, xai_results (+ indexes)
  003_features_table.sql      features (+ index on week)
  004_narratives.sql          narratives table (now dropped by 007)
  005_external.sql            external_signals (week PK + 7 signal cols, index on week)
  006_features_add_external.sql  documentation only — schema change applied via db._ensure_external_cols()
  007_insights.sql            drops narratives; creates insight_findings + insight_summary

data/          M5 raw files (gitignored — downloaded by ingest.py)
external_data/ Committed external-signal CSVs (NOT gitignored — pipeline reads these offline, never the internet)
db/            SQLite databases (gitignored): forecasting.db (production), smoke.db (throwaway)
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
| `xai_results` | run_xai.py | JSON payloads: shap / counterfactual / contrastive |
| `insight_findings` | generate_insights.py | One row per candidate finding: status, confidence, evidence JSON, hypothesis JSON, critic notes |
| `insight_summary` | generate_insights.py | Single 'overall' row: DS-facing + business-facing synthesis JSON, model names |
| `external_signals` | ingest_external.py | Per-fiscal-week LA weather + CA gas + consumer sentiment (Stage 1). NOT yet read by features.py |

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

**Idempotency:** Each script clears its own output tables at start. `backtest.py` clears forecasts + evaluations. `run_xai.py` clears xai_results. `generate_insights.py` clears insight_findings + insight_summary. Re-running any stage produces a clean result with no orphan rows. Downstream stages are safe to re-run without re-running upstream (e.g. re-run `generate_insights.py` alone to fix LLM issues).

**Smoke test isolation:** `smoke_test.py` reads from `db/forecasting.db` (source DB, never written to) and writes all output to `db/smoke.db` (throwaway). Running the smoke test never contaminates dashboard data.

**XAI model — per retrain checkpoint:** SHAP/counterfactual/contrastive for each bad week are computed using the exact retrain checkpoint that produced that week's forecast (at 4-week granularity). `backtest.py` saves each checkpoint to `models/checkpoint_{cutoff}.lgbm` and the week→cutoff mapping to `db/week_to_cutoff.json`. `run_xai.py` loads these from disk so XAI can be re-run independently. Explainers are cached by checkpoint to avoid rebuilding per bad week. Contrastive compares both the bad week and its reference week under the **same model** (the bad week's checkpoint) to keep SHAP profiles in the same space.

**XAI (top 50 worst SKUs per bad week):**
- `shap`: TreeSHAP — top 5 drivers in log-margin space, plus `other_features_shap` (sum of remaining 14) so the waterfall reconciles: `base_value_log + Σ(top5) + other_features_shap ≈ log(prediction)`
- `counterfactual`: zero out SNAP / event / price-change → measure prediction delta. Each scenario includes `was_active: bool` so the dashboard can distinguish meaningful zeroing from a no-op.
- `contrastive`: compare SHAP profile vs a good reference week for the same SKU (same ISO week-of-year, MAPE < 15% in full eval history). Skips items with no same-WOY good week — no fallback to different-seasonality weeks. `contrastive_payloads` must receive `all_evals_df` (full history), not just the current bad week's evals — otherwise good reference weeks are never found. `seasonality_matched: True` in every payload (guaranteed by the skip logic).

**shap_payloads API:** Returns `(list[dict], dict[str, np.ndarray])` — the DB rows and a `shap_cache` mapping uid → raw SHAP array. Pass `shap_cache` to `contrastive_payloads` as `bad_shap_cache` to avoid recomputing bad-item SHAP. Both `backtest.py` and `smoke_test.py` do this.

**SHAP payload extras:** Each SHAP row includes `signed_error` (signed % error, positive = over-forecast) and `direction` ("over"/"under") so the insights module can reference error direction without recomputing it.

**Insights module (`xai_forecast/insights/` + `generate_insights.py`):**
- Replaces the former `narrate.py` / `generate_narratives.py` / `narratives` table.
- Architecture: deterministic detectors fire on real data thresholds → LangGraph `StateGraph` fan-out → per-finding: `run_planner` (Flash, chooses 1-4 read-tools) → `_enrich_evidence` (calls chosen tools) → `run_hypothesis` (Flash) → grounding advisory → `run_critic` (Pro) → fan-in → `run_synthesis` (Flash).
- Two models: `deepseek-v4-flash` for planner/hypothesis/synthesis, `deepseek-v4-pro` for critic. Both via `DeepSeekClient` (OpenAI SDK, base_url `https://api.deepseek.com`). Key via `DEEPSEEK_API_KEY` env var.
- **LLM is mandatory** — `generate_insights.py` fails loudly if `DEEPSEEK_API_KEY` is absent.
- Four `*_PROMPT` constants in `agents.py`: `PLANNER_PROMPT`, `HYPOTHESIS_PROMPT`, `CRITIC_PROMPT`, `SYNTHESIS_PROMPT`. Run `/prompt-audit` before editing any of them.
- **Grounding check is advisory only.** After hypothesis, unmatched evidence refs are forwarded to the Pro critic as `grounding_advisory: {grounding_ok, missing_refs}`. The critic is the single quality gate — the grounding check does not gate or mutate confidence. Tolerant matching: bare leaf names and `[N]`-normalised paths both resolve correctly.
- **Logging:** `generate_insights.py` sets up structured logging (INFO → console, DEBUG → `logs/insights.log`). Every agent step logs at DEBUG. `logs/` is gitignored.
- LangGraph fan-out via `Send()` is serial in sync invocation — SQLite thread safety is not a concern.
- `compute_recurring_drivers(shap_rows)` in `xai_forecast/db.py`: single source of truth for recurring-driver aggregation. Returns `{feature, count, pct_payloads, n_weeks, pct_bad_weeks}` per feature.

**Pre-launch SKUs:** A SKU in its pre-launch weeks has all-NaN lag features. `make_forecasts` imputes these to 0 (via `.fillna(0)` before `model.predict`). If such a SKU has `y > 0` in that week, it is evaluated against a garbage forecast. `backtest.py` counts and logs these rows. `data_quality.py` checks for pre-launch price leakage (sell_price non-null with lag_1 null, excluding the first dataset week — week 1 legitimately has lag_1=NULL for all items since shift(1) returns NaN for the first row; a non-null price there is genuine raw data, not bfill).

## XAI insight quality — known limitations

Findings from full-run analysis (120 backtest weeks, 16 bad weeks, 800 SHAP payloads):

- **100% of bad-week SHAP payloads are over-forecasts.** Bad weeks are always a systematic over-forecast, never under. This is not reflected in the current narratives.
- **Two features dominate everything.** `rolling_4_mean` appears in 91% of payloads, `lag_1` in 87%. The model almost always over-anchors on recent sales history. LLM narratives are generic because the dossier looks the same across most items.
- **Contrastive coverage is 27%** (171 of 645 explained items). Only items with a same-WOY week where MAPE < 15% get contrastive data. 73% of items have no contrastive panel in the dashboard.
- **LLM item narratives add little value** — they paraphrase SHAP feature names without explaining the actual demand event (e.g. "demand collapsed from 65 to 1 unit — demand cliff" vs "recent trend shifted").

**Project reframe (current direction):** the goal is not better forecast accuracy — it is **XAI-driven model governance**. Use the backtest + XAI to produce (a) a data-scientist "what to fix" list and (b) a business-facing "limitations + improvement plan". Model performance is explicitly not the focus; feature engineering stays lean.

**Active plan — external signals:** see [EXTERNAL_SIGNALS_PLAN.md](EXTERNAL_SIGNALS_PLAN.md). Adds a curated fast set of real, committed external signals (LA weather, CA gas price, consumer sentiment) so the XAI has real-world causes to point at instead of only autoregressive lags. Two-stage: Stage 1 = external ingestion only (hard stop + validation gate), Stage 2 = lean feature wiring → backtest → xai → narratives → a new "Model Limitations & Improvement Plan" dashboard view. The full plan, locked decisions, sources, and validation anchors live in that file.

**Stage 1 status: DONE (committed, gate passed).** `external_signals` is populated for all 278 fiscal weeks (2011-01-29 → 2016-05-21), zero gaps. Data is committed real CSVs under `external_data/` (Open-Meteo LA weather, EIA CA gas, U. Michigan consumer sentiment). `validate_external.py` runs 21 checks — all pass, including real-world anchors (Oct-2012 gas spike $4.71, Q1-2016 gas low $2.35, Aug-2011 sentiment 55.7, 2015 sentiment peak 98.1, 2013–15 drought precip below 2011). Gas maps to all 278 weeks with a direct EIA reading (no ffill needed). **Sentiment caveat:** FRED was unreachable from the build machine, so `tools/fetch_external_raw.py` fell back to embedded real historical UMCSENT values (verifiable on FRED); it tries the live endpoint first. **Stage 2 is NOT started** — `features.py`, `FEATURE_COLS`, models, and narratives are all unchanged; the signals are ingested but not yet used by the model.

**Insights module status: LIVE.** `narrate.py` / `generate_narratives.py` / `narratives` table have been removed. The insights module (`xai_forecast/insights/`) is the active LLM layer. See [INSIGHTS_PLAN.md](INSIGHTS_PLAN.md) for the original design rationale. Pipeline step is now `generate_insights.py`; output tables are `insight_findings` + `insight_summary` (migration `007_insights.sql`). `langgraph` dependency added. Dashboard not yet updated to consume the new tables — that is the next planned step.

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
- DeepSeek V4 Flash + V4 Pro via `openai` SDK (insights module — mandatory, fails loudly without key)
- LangGraph (insights orchestration)
