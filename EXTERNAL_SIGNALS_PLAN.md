# External Signals — Implementation Plan

**Status:** **Stage 1 DONE (committed, gate passed). Stage 2 not started.**
**Owner handoff:** Written on Opus (planning). Execution must run on **Sonnet** (project rule: Opus plans, Sonnet executes).
**Last updated:** 2026-06-27

This document is self-contained. A fresh AI session should be able to execute Stage 2 from this file alone.

> **Stage 1 result (do not redo):** `external_signals` populated for all 278 fiscal weeks, zero gaps. Real CSVs
> committed under `external_data/`. `validate_external.py` passes all 21 checks (anchors: Oct-2012 gas $4.71,
> Q1-2016 gas $2.35, Aug-2011 sentiment 55.7, 2015 peak 98.1, drought precip). Gas matched all 278 weeks directly
> (no ffill). **Sentiment note:** FRED was unreachable from the build machine; `tools/fetch_external_raw.py` fell
> back to embedded real historical UMCSENT values (it tries live FRED first). Nothing downstream was touched —
> `features.py`, `FEATURE_COLS`, models, narratives are all unchanged. **Next: Stage 2 (section 5).**

---

## 1. Why we're doing this (the reframe)

The project's purpose has shifted. It is **not** "build a better forecaster." It is a demonstration of
**XAI-driven model governance** — using backtesting + explainability to produce two concrete outputs:

- **For the data scientist:** a prioritized "what to fix" list — where and why the model fails, which signals it lacks.
- **For the business:** a plain-language account of the model's limitations *and* the roadmap to improve it.

**Model accuracy is explicitly NOT the focus.** External signals are added to give the XAI *something real to
point at* and to make the "here's what's missing / here's the plan" story credible — not to chase WMAPE.
Keep feature engineering deliberately lean.

### The two key questions the project must answer

- **Data scientist:** Does the model + XAI forecast well long-term, and what should we fix?
- **Business:** Does it work, what's the ROI, what's the risk, is it scalable — and what's the plan to improve?

---

## 2. Background findings that motivated this (from the full backtest run)

Full run = 120 backtest weeks, 16 bad weeks, 800 SHAP payloads. Analysis of the stored XAI showed:

- **100% of bad-week explanations are over-forecasts.** Bad weeks are always a systematic over-forecast, never under.
- **Two features dominate every explanation:** `rolling_4_mean` (91% of payloads), `lag_1` (87%). The model is
  essentially autoregressive.
- **Contrastive coverage is only 27%** (171 of 645 explained items) — most items have no same-week-of-year good reference.
- **LLM item narratives are generic** — they paraphrase SHAP feature names without explaining the actual demand event.

**Root cause:** this is a **data ceiling, not a prompt problem.** M5 (the dataset) contains almost no causal
features — only SNAP, event flag, price, week-of-year as exogenous signals. SHAP can only ever point back at the
autoregressive lags because there are no real-world causes in the table. The fix is to add real causes.

### Dataset decision

We evaluated switching datasets (Favorita, Rossmann, Walmart Recruiting, dunnhumby). **Decision: stay on M5** and
augment it in place with external data. Rationale: keeps the existing pipeline and benchmark, lowest rework, and
M5 + weather augmentation is a recognized accuracy/explainability boost. (If the external-signal experiment proves
the thesis but still feels thin, Favorita is the documented fallback — it has built-in promotions + perishable flag.)

### Dataset geography (confirmed from the data)

- **Country:** United States. **State:** California. **Store:** `CA_1` (every SKU id ends in `_CA_1`; the SNAP
  column is California's CalFresh schedule).
- **Window:** 2011-01-29 → 2016-05-21 (Saturday-stamped fiscal weeks).
- **Granularity ceiling:** Walmart anonymized M5 to **state level** — no city, no coordinates. All external joins
  are at **California-state resolution**, keyed by date → fiscal week. Weather uses a documented city proxy (see below).

---

## 3. Locked decisions

| # | Decision | Value | Notes |
|---|---|---|---|
| 1 | Dataset | Stay on M5 (CA_1) | Augment in place |
| 2 | Signal timing | **Contemporaneous** (current-week external) | Correct for retrospective explainability. Document as a mild lookahead in the `features.py` leakage section. (Lagged is the operational alternative if ever needed.) |
| 3 | Weather proxy | **Los Angeles single point** (≈34.05, -118.24) | Stated assumption — CA is climatically split; LA is the largest population center. Population-weighted multi-city is a later upgrade. |
| 4 | Data source | **Real, fetched once, committed as static CSV** | Pipeline reads the committed file offline. No live APIs in the pipeline, no keys at run time. NOT LLM-generated values — those would be fabricated and would poison every downstream explanation. |
| 5 | Signal scope | **Curated fast set** | Weather + CA gas + consumer sentiment. The signals with sharp, explainable per-week effects. Skip slow/collinear macro (oil, unemployment, CPI, income). |
| 6 | Deliverable | **New dashboard view**: "Model Limitations & Improvement Plan" | DS "what to fix" section + business "limitations + roadmap" section. Lives in the existing Streamlit app alongside the current 4 pages. |
| 7 | Execution | **Two stages with a review gate** | Stage 1 = external ingestion only, hard stop. Pause + review against success criteria. Then Stage 2 = everything else. |

---

## 4. STAGE 1 — External signal ingestion (build this first, then STOP)

### 4.1 Columns (7 signals)

Table `external_signals`, one row per fiscal week:

```
external_signals(
  week TEXT PRIMARY KEY,   -- Saturday fiscal-week date, joins to features.week
  temp_mean REAL,          -- weekly mean of daily mean temp (°C)
  temp_max  REAL,          -- weekly max of daily max temp (°C)
  temp_min  REAL,          -- weekly min of daily min temp (°C)
  precip    REAL,          -- weekly total precipitation (mm)
  heat_days INTEGER,       -- count of days in the week with daily max > 32°C (~90°F)
  gas_price REAL,          -- CA regular retail gasoline ($/gal)
  consumer_sentiment REAL  -- U. Michigan consumer sentiment index
)
```

### 4.2 Sources (real data, fetched once)

| Signal | Source | Key? | Native cadence | Alignment to fiscal week |
|---|---|---|---|---|
| Weather (temp mean/max/min, precip, heat_days) | Open-Meteo Historical/Archive API, LA coords | None | Daily | Aggregate **Sat→Fri**, stamp with the Saturday date |
| CA gas price | EIA weekly California gasoline (bulk CSV download — no key needed) | None (bulk CSV) | Weekly (Monday-stamped) | Map each EIA week into the fiscal week it falls in (nearest / forward-fill) |
| Consumer sentiment | U. Michigan via FRED series `UMCSENT` (CSV download) | None (CSV) | Monthly | Forward-fill the monthly value across the weeks of that month |

### 4.3 Build steps

1. **`migrations/005_external.sql`** — create `external_signals` (schema above) + index on `week`.
   (Schema auto-applies via `get_conn()` → `_setup_schema()`. Never edit existing migrations.)
2. **Run-once fetch script** (e.g. `tools/fetch_external_raw.py`) — hits the three real sources for the
   2011-01-29 → 2016-05-21 window, writes raw CSVs, and we **commit** them.
   - **Gitignore note:** `data/` is gitignored (M5 raw files). Committed external CSVs must live in a **new tracked
     directory** — use `external_data/` at repo root (NOT under `data/`).
   - This script touches the internet exactly once. Everything the pipeline depends on is the committed CSV.
3. **`ingest_external.py`** — reads the committed CSVs, performs the Sat→Fri alignment, writes `external_signals`.
   Idempotent: clears + rewrites its own table at start (matches the project's per-stage idempotency convention).
4. **Validation report** — runs the anchor + implicit checks below, prints a PASS/FAIL report (data_quality style).
   Must **fail loudly** before bad/fabricated data is trusted.
5. **HARD STOP.** Do NOT touch `features.py`, `FEATURE_COLS`, models, narratives, or any downstream stage.

### 4.4 Validation anchors (the data-integrity guardrail)

The committed data MUST reproduce these known real-world facts. This is what catches a wrong region, wrong units,
or fabricated values. Ranges are intentional where exact figures aren't certain — do not assert fake precision.

**Explicit anchors (specific known events/values):**

| Signal | Expectation | Confidence |
|---|---|---|
| CA gas | Local spike to ~$4.6–4.7 around **Oct 2012** (CA refinery crisis) | High |
| CA gas | Declines from ~$4 (mid-2014) to ~$2.5–2.8 (early 2016) during the oil crash | High |
| Consumer sentiment | Sharp drop to ~**55–60 in Aug 2011** (debt-ceiling crisis); recovers to ~90+ by 2015 | Med-High |
| Weather (LA) | **Drought 2012–2016** → below-normal precip; summer (Jul–Sep) weeks warmest | High |

**Implicit checks (structural sanity):**

- Ordering: `temp_max ≥ temp_mean ≥ temp_min` every week; `precip ≥ 0`; `heat_days ∈ [0, 7]`.
- Ranges: `gas_price ∈ [2.0, 5.0]`; LA `temp_mean ∈ [8, 30]` °C; `consumer_sentiment ∈ [55, 100]`.
- Seasonality: `temp_mean` correlates with month-of-year (summer peak). A flat series = a constant got fetched by mistake.
- Coverage: every fiscal week 2011-01-29 → 2016-05-21 present, no gaps, no all-null or all-zero columns post-fill.

Explicit + range checks run **at fetch time** (fail before committing). Structural/coverage checks also get added
to `data_quality.py` so they re-verify on every later pipeline run.

### 4.5 Stage 1 success criteria (the gate to Stage 2)

- [x] `external_signals` populated, **one row per fiscal week**, full coverage 2011-01-29 → 2016-05-21, zero gaps.
- [x] Real data **committed as CSV** under `external_data/` (version-controlled, offline-reproducible).
- [x] **All explicit anchors pass.**
- [x] **All implicit checks pass.**
- [x] Printed PASS/FAIL validation report exists (`validate_external.py`, 21 checks).
- [x] **Nothing downstream touched** — `features.py`, `FEATURE_COLS`, models, narratives all unchanged.

All boxes green (commit `64f3053`). **Stage 1 gate passed — awaiting human review before Stage 2.**

---

## 5. STAGE 2 — Everything else (outline; detail after Stage 1 review)

Do not start until Stage 1 is reviewed and signed off. Keep feature wiring **lean** (decision 5).

1. **`features.py`** — LEFT JOIN `external_signals` on `week`; add the 7 curated columns to `FEATURE_COLS`
   (~19 → ~26). Document the contemporaneous lookahead (decision 2) in the leakage-controls section.
2. **Rebuild feature store** — `build_features.py` (always clears + rebuilds).
3. **Smoke test** — `smoke_test.py` (catches feature staleness via live diff).
4. **`backtest.py`** — retrains. NOTE: the old `models/` checkpoints become invalid the moment the feature schema
   changes; backtest rewrites them, and `run_xai.py` depends on the new ones.
5. **`run_xai.py`** — new features automatically enter SHAP (XAI reads `FEATURE_COLS`; no XAI code change needed).
6. **`narrate.py` dossier update → `generate_narratives.py`** — surface weather/gas/sentiment in plain language
   ("a heatwave", not "temp_max=34"). **This is the payoff step — without it the new signals won't appear in any narrative.**
   Note: `*_PROMPT` constant edits require `/prompt-audit` before commit (project rule).
7. **`data_quality.py`** — add external coverage checks (no post-fill nulls, week-count match, alignment sanity).
8. **Tests** — fix hardcoded feature-count assertions (currently 19); add a join/alignment test for the external data.
9. **New dashboard view** — "Model Limitations & Improvement Plan" page in `app.py`:
   - **DS "what to fix":** over-forecast bias, autoregressive over-anchoring, contrastive coverage gap, which
     external signals do/don't help (feature ablation framing).
   - **Business "limitations + roadmap":** plain-language model limits, risk register (demand shocks, promo blind
     spots), and the improvement plan.
10. **Docs** — update `CLAUDE.md` + `AGENTS.md`: new stage, new table, new feature group, leakage note. Run a
    conflict pass (don't just append).

### Stage 2 notes / candidate improvements (from background findings)

- Consider **failure-pattern classification** before the LLM call (`demand_cliff`, `demand_spike`, `price_driven`,
  `seasonal_drift`) to make narratives targeted instead of generic.
- Consider **deterministic item narratives** (template with concrete numbers), reserving the LLM for week + executive scope.
- Consider widening **contrastive** matching from exact same-WOY to ±2 weeks to raise coverage from 27%.
- Caveat to set expectations: even with external signals, lag features stay strong (demand is autocorrelated). The
  explainability win comes mostly from the **fast** signals (weather, gas, sentiment shocks), not slow macro context.

---

## 6. Project conventions a follow-up session must respect

- **Model rule:** Opus plans only; Sonnet executes/writes code. Surface and switch if mismatched.
- **Always commit AND push** after every change; **check CI** (`gh run list --limit 1`) — note: this repo currently
  has no CI workflows configured.
- **Windows / cp1252:** no non-ASCII in `print()` statements (use `->` not `→`, `-` not `─`). This has bitten the
  project repeatedly.
- **Long-running commands:** use PowerShell `run_in_background`, not Bash `&` (SIGHUP kills it). Avoid `2>&1` with
  native exes in PowerShell 5.1.
- **Each pipeline stage is independently re-runnable** and clears only its own output tables.
