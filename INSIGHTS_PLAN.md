# Insights Module Plan — Evidence-First XAI Synthesis

Status: **planned, not started.** This document is the decision record for replacing the
LLM narrative layer with an evidence-first "insights" module. It captures the *why* and the
*shape*, not line-by-line implementation. Read this before building or changing the module.

---

## 1. Why we are doing this

### The founding question
The project exists to answer one question for a business leader:

> "The model performed badly at week X — **why?**"

### The reframe (already locked in CLAUDE.md)
The goal is **not** better forecast accuracy. It is **XAI-driven model governance**: use the
backtest + XAI to produce (a) a data-scientist "what to fix" list and (b) a business-facing
"limitations + improvement plan". Feature engineering stays lean. Model performance is
explicitly not the focus.

### What is wrong with the current narrative layer
The existing `narrate.py` / `generate_narratives.py` layer has known, documented weaknesses:

- **Generic output.** Two features (`rolling_4_mean` 91%, `lag_1` 87%) dominate every bad
  week, so the dossier looks the same each time and the LLM paraphrases SHAP feature names
  ("recent sales trend changed") instead of describing the real demand event.
- **It describes what the model *looked at*, not what *happened in the world*.** A leader
  reads "recent trend shifted" and learns nothing actionable.
- **Weak guard against fabrication.** A single grounding check (is `primary_driver` a known
  feature?) — no defense against invented causes, especially once external signals are in play.
- **Buries the most important finding.** Every flagged bad week is an *over-forecast* (the
  model over-anchors on recent sales momentum and keeps predicting high after demand drops).
  This systematic bias — the single most useful governance insight — is never surfaced.

The fix is not a better prompt. It is a **different architecture**: detect what went wrong,
prove it from the data, and only then let the LLM narrate it.

---

## 2. The pattern we are copying — and the principle behind it

### Reference project
`dfs-ai-cost-analysis` (in the same playground folder) — an evidence-first cost-leakage
analyzer. Its design doc (`docs/agent_walkthrough.md`) is the canonical reference.

### What makes it good
- **Code owns the evidence.** Deterministic detectors scan the data and emit concrete
  *candidate findings*. Each becomes a compact **evidence pack** traceable to row-level source.
- **The LLM only interprets bounded evidence**, never the raw dataset.
- **A critic step rejects overclaim** and downgrades weak-confidence findings.
- **LangGraph orchestrates**, but the graph is small and bounded (mostly linear + one fan-out).
- It explicitly *avoids*: "full-dataset freeform prompting, **taxonomy-first dependence**,
  opaque agent-only reasoning."

### Evidence-first vs taxonomy-first (the crux)
- **Taxonomy-first** = predefine abstract failure categories, then force the data to fit them.
- **Evidence-first** = deterministic detectors fire on real thresholds in the data, carry the
  actual rows forward, and the LLM only interprets what is already grounded.

Our detectors *look* like a taxonomy (demand_cliff, over_forecast_bias, …) but they are
**evidence triggers**: each fires only when a concrete threshold is crossed in real data and
carries the supporting rows. That keeps us on the evidence-first side of the line. This
distinction is the whole reason this approach beats a multi-chain LLM workflow.

---

## 3. Target architecture

```
detectors (deterministic) → candidate findings        ← evidence triggers, traceable
   ↓  fan-out per finding
planner-executor agent (deepseek-v4-flash + READ TOOLS)
   plan → call tools to gather evidence → hypothesis    ← agent PULLS context, not fed a dossier
   ↓
critic (deepseek-v4-pro) → reject untraceable / causal-external claims, set confidence
   ↓  fan-in accepted findings
synthesis (deepseek-v4-flash) → { data_scientist, business_leader }
```

### Key architectural decisions
- **LLM is mandatory.** `generate_insights.py` requires `DEEPSEEK_API_KEY` and fails loudly if
  missing. No detectors-only / no-LLM output mode. (Detectors still run *inside* the graph, but
  the stage produces nothing without the LLM.)
- **The agent has read-tools, not a pre-built dossier.** It actively pulls internal signals,
  external signals, forecasts, good/bad weeks, and XAI findings on demand. This is a true
  planner-executor, richer than the reference's fixed evidence packs.
- **The critic runs on the Pro model** — it is the quality gate, worth the cost. Hypothesis and
  synthesis run on Flash.
- **Bounded.** Detectors limit the candidate set (~6 finding types); fan-out is capped. The
  graph stays small — we are not building an open-ended research agent.

### Read-tools the agent can call
| Tool | Reads from | Returns |
|---|---|---|
| `read_forecast_accuracy()` | evaluations, forecasts | weekly MAPE series + bad/good flags |
| `read_bad_weeks()` / `read_good_weeks(item)` | evaluations | flagged weeks / good reference weeks |
| `read_xai_findings(week, item)` | xai_results | SHAP / counterfactual / contrastive payloads |
| `read_demand_trajectory(item, week)` | weekly_sales, features, forecasts | prior sales → prediction → actual |
| `read_external_signals(week)` | external_signals | weather / gas / sentiment for that week |
| `read_model_metadata()` | checkpoints + week_to_cutoff.json | feature importance, training window, retrain cadence |
| `read_recurring_drivers()` | xai_results | driver appearance frequency across bad weeks |

### Detectors (evidence triggers — starting set)
`over_forecast_bias`, `dominant_driver`, `demand_cliff`, `external_coincidence`,
`counterfactual_material`, `contrastive_gap`. Each fires on a concrete data threshold and
carries the supporting rows. The list can grow; it should never become an imposed taxonomy.

### Critic discipline (the core guard)
Rejects any claim not traceable to the evidence; **external signals may only be stated as
correlation, never cause**; downgrades confidence when evidence is thin. This is what kills the
generic-narrative problem and keeps the output defensible for governance.

---

## 4. Model assignment

| Step | Model | Why |
|---|---|---|
| Hypothesis (per finding) | `deepseek-v4-flash` | High volume, cheap |
| Critic (per finding) | `deepseek-v4-pro` | Careful overclaim guard — the quality gate |
| Synthesis (overall) | `deepseek-v4-flash` | One call, structured |

base_url `https://api.deepseek.com`. Env: `DEEPSEEK_MODEL` (default `deepseek-v4-flash`),
`DEEPSEEK_CRITIC_MODEL` (default `deepseek-v4-pro`), `DEEPSEEK_API_KEY` (required).
Model names verified against DeepSeek's official pricing docs.

---

## 5. What we touch

| Stage | File | Status | Note |
|---|---|---|---|
| Ingest M5 | `ingest.py` | Untouched | Raw tables |
| External signals | `ingest_external.py`, `validate_external.py` | Untouched | Stage 1, done |
| Build features | `build_features.py` | Untouched | 26-feature store rebuilt |
| Smoke test | `smoke_test.py` | Change | Repoint live-API probe → insights |
| Backtest | `backtest.py` | Untouched | Already re-run (18 bad weeks) |
| XAI | `run_xai.py`, `xai_forecast/xai.py` | Untouched | `xai_results` is the evidence base |
| ~~Narratives~~ | `generate_narratives.py`, `xai_forecast/narrate.py` | **REMOVE** | Replaced wholesale |
| **Insights (new)** | `generate_insights.py` + `xai_forecast/insights/` | **NEW** | Evidence-first planner-executor |
| Data quality | `data_quality.py` | Change | Swap narrative checks → insight-table checks |
| Dashboard | `app.py` | Rewrite | 4 confusing pages → 1 clean page |
| Code walkthrough | `code_review.py` | Untouched | Left alone |
| DB helpers | `xai_forecast/db.py` | Change | Remove narrative helpers, add insight helpers |
| Schema | `migrations/007_insights.sql` | **NEW** | Drop `narratives`; create insight tables |
| Dependency | `pyproject.toml` | Change | `uv add langgraph` |

**New pipeline order:** `… backtest → run_xai → generate_insights → data_quality → app`

Only one command name changes: `generate_narratives.py` → `generate_insights.py`.

### Cleanup (deleted)
- `xai_forecast/narrate.py`, `generate_narratives.py`, `tests/test_narrate.py`
- `narratives` table (dropped in migration 007 — migration 004 stays on disk; never edit old migrations)
- Narrative helpers in `db.py`; narrative probe in `smoke_test.py`; narrative pages in `app.py`
- Doc references in `CLAUDE.md`, `AGENTS.md`, `EXTERNAL_SIGNALS_PLAN.md`

### New output tables (migration 007)
- `insight_findings` — the auditable ledger: finding_id, type, week_id, item_id, evidence JSON,
  hypothesis, status (accepted/rejected/needs_review), confidence, created_at
- `insight_summary` — the overall two-perspective summary: key=`overall`, `data_scientist` JSON,
  `business_leader` JSON, model, created_at

---

## 6. Old vs new

| | Old (narratives) | New (insights) |
|---|---|---|
| Trigger | Loops every bad week, always calls LLM | Detectors fire only on real thresholds |
| What LLM sees | Pre-shaped dossier of SHAP names | Evidence pulled via tools, traceable to rows |
| Anti-fabrication | One grounding check | Critic on Pro model; external = correlation only |
| Output | 3 scopes, generic | 1 overall 2-perspective summary + findings ledger |
| Failure honesty | Over-forecast bias hidden | Each claim cites a specific finding with numbers |
| Orchestration | Linear Python loop | Small bounded LangGraph |
| Storytelling | "recent sales trend changed" | "sold ~58/wk then cliffed to 4; model predicted 57 because it anchors on recent weeks" |

Core shift: **from "describe what the model looked at" to "detect what went wrong, prove it,
then narrate it."**

---

## 7. Locked decisions

- **Summary granularity:** overall only (one DS + one business summary). Per-week/item detail
  lives in the findings ledger / drill-down, not as separate summaries.
- **Dashboard:** single scrolling page — MAPE chart (top) → two-perspective summary (middle) →
  findings-ledger drill-down (bottom, with SHAP/counterfactual/contrastive kept).
- **XAI types:** keep all three in the drill-down.
- **Agent depth:** full — detectors → packs → critic → synthesis. (Not the lighter no-critic
  version; not the heaviest web-lookup/feedback-file version.)
- **LangGraph:** yes, added as the orchestrator.
- **LLM:** mandatory.
- **`code_review.py`:** left untouched.

---

## 8. Success criteria

### Functional
1. `generate_insights.py` runs green; populates `insight_findings` + `insight_summary`;
   re-runnable (clean-slate, no orphans).
2. All old narrative code + `narratives` table gone; `pytest` green (incl. `test_insights.py`).
3. `data_quality.py` passes including new insight-table checks.
4. Single-page dashboard renders summary + ledger drill-down; no dead pages.

### Quality (the actual point)
5. **Traceability** — every ledger finding links back to specific `xai_results` / sales rows;
   nothing in the business summary unbacked by a finding.
6. **No overclaim** — the critic demonstrably rejects/downgrades; external signals never stated
   as causes.
7. **Two real audiences** — DS summary names concrete fixes (e.g. "model over-anchors on
   rolling_4_mean; metric is blind to under-forecasts"); business summary states risk direction
   (over-forecast → over-stock), limitations, and an improvement plan.
8. **Specificity** — at least demand_cliff and over_forecast_bias findings carry real numbers,
   not feature-name paraphrase.

### Governance framing
9. The output answers "Leader sees the model performed badly at week X — why?" in a way both a
   data scientist and a business leader can act on.

---

## 9. Known tradeoffs / risks

- **More moving parts** than a single dossier→call. Mitigation: keep the graph small and bounded.
- **New dependency (langgraph).** Accepted — it is the orchestrator the reference uses.
- **Cost.** Critic on Pro per finding costs more; bounded by the small number of finding types.
- **Over-forecast bias must be re-confirmed on the new 18-bad-week run** before the executive
  story leans on it — it was measured on the old 19-feature run. The `direction` field already
  exists in the SHAP payloads; a detector (`over_forecast_bias`) computes it from live data, so
  the story stays grounded in the current run rather than an assumption.
- **Prompt constants** (`HYPOTHESIS_PROMPT`, `CRITIC_PROMPT`, `SYNTHESIS_PROMPT`) must pass
  `/prompt-audit` before commit, per project rule.

---

## 10. Execution note

All of this is execution work (deletes, new subpackage, LangGraph wiring, dashboard rewrite,
prompts + audit). Per the project model rule, it must be built on **Sonnet** — Opus plans only.
Suggested order: cleanup + dependency + schema/detectors → read-tools + graph + critic →
synthesis + runner → dashboard rewrite → data_quality + tests → docs. Commit and push at each
stable checkpoint; this repo has no CI configured.
