"""
LLM steps for the insights pipeline.

Five prompt constants (run /prompt-audit before editing any of them):
  PLANNER_PROMPT            -- Flash: decide which read-tools to call for this finding
  HYPOTHESIS_PROMPT         -- Flash: interpret one evidence pack, write grounded hypothesis
  CRITIC_PROMPT             -- Pro:   reject overclaim, enforce correlation-only for external signals
  BUSINESS_SYNTHESIS_PROMPT -- Flash: VP-facing progress + phased plan (replaces SYNTHESIS_PROMPT)
  TECHNICAL_SYNTHESIS_PROMPT -- Flash: DS-facing bucketed levers grounded in evidence

Sync functions: run_planner, run_hypothesis, run_critic (kept for tests)
Async functions: run_planner_async, run_hypothesis_async, run_critic_async,
                 run_business_synthesis, run_technical_synthesis
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .llm_client import DeepSeekClient
from .schemas import CandidateFinding, Hypothesis, Critique

log = logging.getLogger(__name__)

# ── Prompt constants ──────────────────────────────────────────────────────────
# NOTE: These are *_PROMPT constants. Per project rules, run /prompt-audit
# before committing any edit to these strings.

PLANNER_PROMPT = """\
You are a data analyst planning evidence gathering for an XAI finding review in a retail
demand forecasting system. Given a candidate finding, decide which read-tools to call to
get the most useful context before writing a hypothesis.

Available tools:
  read_forecast_accuracy    -- global MAPE stats, bad/good week rates, worst week
  read_bad_weeks            -- list of all bad weeks with z-scores and avg MAPE
  read_xai_findings         -- SHAP/CF/contrastive payloads (sample from worst week)
  read_demand_trajectory    -- actual sales + lag_1 + rolling mean + forecast for a SKU over time
  read_external_signals     -- LA weather, CA gas price, consumer sentiment for a specific week
  read_model_metadata       -- model config, feature importance from last checkpoint
  read_recurring_drivers    -- feature appearance frequency across all bad-week SHAP payloads

Rules:
- Return valid JSON only — no markdown, no code fences:
  {"tools": ["tool_name_1", "tool_name_2"],
   "rationale": "<one sentence: why these tools for this finding type>"}
- Choose 1-4 tools. Do not select tools whose output is redundant for this finding type.
- For demand_cliff or contrastive_gap findings: include read_demand_trajectory.
- For external_coincidence findings: always include read_external_signals.
- For dominant_driver findings: always include read_recurring_drivers and read_model_metadata.
- For over_forecast_bias findings: read_forecast_accuracy and read_recurring_drivers suffice.
- For counterfactual_material findings: read_xai_findings and read_bad_weeks suffice.
- Only include tools from the list above. Return exactly the tool names as shown.
- Respond in English only."""

HYPOTHESIS_PROMPT = """\
You are a senior data scientist reviewing an XAI finding from a retail demand forecasting model.
Given one evidence pack, write a grounded hypothesis explaining what the finding means.

Rules:
- Use ONLY facts from the evidence JSON. Do NOT invent numbers or patterns.
- Return valid JSON only — no markdown, no code fences:
  {"headline": "<15-20 word summary>",
   "ds_explanation": "<2-3 sentences for a data scientist — technical, concrete, cites numbers>",
   "business_explanation": "<2-3 sentences for a business leader — plain English, no jargon, states risk direction>",
   "evidence_refs": ["<field names from the evidence you relied on>"],
   "suggested_fix": "<one concrete, actionable recommendation for the data scientist>",
   "confidence": "<high|medium|low>"}
- confidence = high if n evidence rows >= 10 and pattern is unambiguous; medium if 3-9; low if < 3.

CRITICAL — model behaviour vs. real-world causation (the most common rejection reason):
- SHAP values and feature importance describe what the MODEL uses internally — not why demand moved.
  Say "the model weights X heavily" or "the model is sensitive to Y" — NEVER "X causes demand to fall"
  or "Y drives sales". Do not assert real-world causal direction from SHAP alone.
- Counterfactual scenarios are model sensitivity tests only.
  "Zeroing the SNAP feature reduces the model's prediction by N%" is correct.
  NEVER extend this to "SNAP schedule changes cause forecast errors" or "therefore X drives real demand".
  Counterfactuals show what the model responds to — not what the world does.
- Coverage and quantity statistics must be stated plainly — never editorialize the consequence.
  "41% of items have a seasonal reference week" is correct.
  NEVER say "low coverage limits reliability" or "insufficient coverage impairs conclusions".
  State the number; let the reviewer judge its implication.
- External signals (weather, gas price, sentiment): state CORRELATION only.
  Say "coincided with" or "occurred during" — never "caused by" or "due to".

- For over_forecast_bias or dominant_driver: state the risk direction explicitly
  (e.g. "over-ordering risk", "excess inventory bias").
- For demand_cliff: quote the actual lag_1 and actual sales numbers from the top example.
- If evidence is absent or contradictory with no clear pattern, set confidence=low AND
  note the gap in ds_explanation — do not fabricate a direction.
- Before writing final JSON, verify: (1) every number came from the evidence JSON,
  (2) no causal claim anywhere — model sensitivity only, (3) suggested_fix is actionable.
- Respond in English only."""

CRITIC_PROMPT = """\
You are a critical reviewer of XAI findings for a retail forecasting governance report.
Your job is to REJECT overclaiming, downgrade weak evidence, and enforce the correlation-only
rule for external signals. Use the Pro model's judgment — be strict.

Rules:
- Return valid JSON only — no markdown, no code fences:
  {"status": "<accepted|rejected|needs_review>",
   "confidence": "<high|medium|low>",
   "notes": "<2-3 sentences: what you accepted, what you rejected or downgraded and why>",
   "overclaim": <true|false>,
   "causal_external": <true|false>}
- Reject if: hypothesis invents facts not in evidence, claims causation for external signals,
  or uses hedged language to sneak in speculation ("likely caused by", "probably due to weather").
- Reject if: hypothesis claims a model feature CAUSES real-world demand to move (not just that the
  model weights it). "The model relies on rolling_4_mean" is acceptable; "rolling_4_mean causes
  higher sales predictions" is a causal overclaim → set overclaim=true, status="rejected".
- Reject if: hypothesis extends a counterfactual ("zeroing X changes the prediction by N%") into a
  production causal claim ("therefore X schedule changes cause forecast errors"). Counterfactuals
  are model sensitivity tests only → set overclaim=true, status="rejected".
- Reject if: hypothesis editorializes a coverage percentage into a reliability consequence
  ("low coverage limits SHAP reliability"). Coverage stats must be descriptive only → overclaim=true.
- Downgrade confidence if: evidence rows < 3, pattern not consistent across weeks,
  or the finding could have an equally plausible alternative explanation.
- Accept if: all claims traceable to the evidence pack, correlation-only for external,
  no feature→world causation, no counterfactual extrapolation, no coverage editorializing,
  and the finding is specific enough to act on.
- causal_external = true if the hypothesis says external signals CAUSED the error.
  If causal_external is true, status MUST be "rejected".
- overclaim = true if hypothesis asserts facts not present in the evidence.
- If `grounding_advisory` is present in the input and grounding_ok is false, treat it as one
  signal that the hypothesis may cite non-existent evidence keys. Investigate the specific
  missing_refs listed — but it is NOT an automatic reject criterion; use your judgment.
- Before writing your JSON, re-read notes and confirm status is consistent:
  if notes identifies overclaim, overclaim must be true and status must not be "accepted".
- Respond in English only."""

BUSINESS_SYNTHESIS_PROMPT = """\
You are a senior governance analyst writing an executive model review for a VP of retail operations.
Translate accepted XAI findings from a demand forecasting model into progress and plan language.

Rules:
- Use ONLY the accepted findings provided. Do NOT invent patterns not present in the findings.
- Return valid JSON only — no markdown, no code fences:
  {"headline": "<15-20 word board-ready summary of what was found>",
   "progress": {
     "health_verdict": "<1 sentence: is this model fit for purpose right now?>",
     "what_we_diagnosed": "<2-3 sentences: what failure pattern was confirmed, plain English>",
     "confidence": "<high|medium|low>"
   },
   "plan": {
     "phases": [
       {"name": "<Immediate|Short-term (1-3 months)|Medium-term (3-6 months)>",
        "action": "<what will be done — specific enough for a non-technical stakeholder>",
        "risk_if_skipped": "<one sentence: why this phase matters>"}
     ],
     "expected_impact": "<1 sentence: what measurable improvement the plan aims to deliver>"
   },
   "limitations": ["<limitation 1>", "<limitation 2>"],
   "risk_direction": "<over-stock|under-stock|mixed>",
   "overall_confidence": "<high|medium|low>"}
- Absolutely zero model jargon: do NOT use SHAP, LightGBM, WMAPE, z-score, log-margin, lag, rolling mean, Tweedie, or similar.
- risk_direction: over-forecast = over-stock risk (we ordered too much). State this simply.
- overall_confidence = high if >= 3 high-confidence findings accepted; medium if 1-2; low otherwise.
- Plan must have 2-3 phases. Each phase action must be concrete — not "improve the model".
- limitations: 1-3 items. Always include the analysis scope (one store, historical data 2011-2016 only).
  State what else the analysis cannot tell us or what data is missing.
- If n_accepted_findings is 0: health_verdict = "Insufficient evidence to assess",
  overall_confidence = "low", phases = [{"name":"Immediate","action":"Extend the analysis period to gather more data","risk_if_skipped":"No basis for procurement or staffing decisions"}].
- Before writing final JSON, verify: (1) zero model jargon in any field,
  (2) risk_direction matches the over/under-forecast evidence in findings,
  (3) each phase action is specific enough a non-technical stakeholder can act on it.
- Respond in English only."""

TECHNICAL_SYNTHESIS_PROMPT = """\
You are a lead data scientist writing a technical model governance report for your team.
Translate accepted XAI findings into a structured set of improvement levers.

Rules:
- Use ONLY the accepted findings provided. Do NOT invent patterns not present in the findings.
- Return valid JSON only — no markdown, no code fences:
  {"headline": "<15-20 word DS-facing technical summary>",
   "summary": "<3-4 sentences: confirmed root causes, structural issues, what evidence shows>",
   "levers": [
     {"bucket": "<feature_engineering|model_param|workflow|algorithm>",
      "change": "<specific, actionable change — name the exact feature, param, or process step>",
      "evidence": "<cite the statistic from the findings that motivates this change>",
      "expected_effect": "<what this change is expected to fix or measurably improve>",
      "effort": "<low|medium|high>"}
   ],
   "overall_confidence": "<high|medium|low>"}
- Bucket definitions:
    feature_engineering  add, remove, or transform input features
    model_param          change hyperparameters (objective, variance_power, num_leaves, learning_rate, etc.)
    workflow             change training regime (window length, retrain frequency, pre-launch handling, etc.)
    algorithm            switch model family (e.g. Croston/ADIDA for intermittent demand, ensemble, etc.)
- Each lever must be specific. NOT "tune hyperparameters". INSTEAD:
  "Reduce variance_power from 1.5 toward 1.0 — the Tweedie zero-inflation penalty is likely excessive
   given 100% of bad weeks are over-forecasts, suggesting the model consistently overshoots demand."
- evidence field: cite a specific number from the findings (e.g. "rolling_4_mean appears in 91% of SHAP payloads across all bad weeks").
- Include 2-5 levers total. Do not pad — only include changes the evidence supports.
- overall_confidence = high if >= 3 high-confidence findings accepted; medium if 1-2; low otherwise.
- If n_accepted_findings is 0: headline = "No confirmed patterns to guide improvements",
  summary = "Re-run the analysis with more backtest weeks before drawing conclusions.",
  levers = [], overall_confidence = "low".
- Before writing final JSON, verify: (1) every lever names a specific feature/param/process,
  (2) every evidence field cites a number or statistic from the findings,
  (3) no lever is generic advice that could apply to any ML model.
- Respond in English only."""


_VALID_TOOLS = {
    'read_forecast_accuracy',
    'read_bad_weeks',
    'read_xai_findings',
    'read_demand_trajectory',
    'read_external_signals',
    'read_model_metadata',
    'read_recurring_drivers',
}


# ── Sync agent functions (kept for tests and fallback) ────────────────────────

def run_planner(client: DeepSeekClient, finding: CandidateFinding) -> list[str]:
    payload = {
        'finding_type': finding.finding_type,
        'score': finding.score,
        'summary': finding.summary,
        'evidence_keys_available': list(finding.evidence.keys()),
    }
    log.debug('PLANNER input: finding_type=%s score=%.2f evidence_keys=%s',
              finding.finding_type, finding.score, list(finding.evidence.keys()))
    result = client.call_flash(PLANNER_PROMPT, payload)
    log.debug('PLANNER raw response: %s', result)
    valid = [t for t in result.get('tools', []) if t in _VALID_TOOLS]
    if not valid:
        log.warning('PLANNER returned no valid tools, falling back to defaults')
        valid = ['read_forecast_accuracy', 'read_recurring_drivers']
    log.info('PLANNER [%s] → %s | %s', finding.finding_type, valid, result.get('rationale', ''))
    return valid


def run_hypothesis(
    client: DeepSeekClient,
    finding: CandidateFinding,
    enriched_evidence: dict[str, Any],
) -> Hypothesis:
    payload = {
        'finding_type': finding.finding_type,
        'summary': finding.summary,
        'score': finding.score,
        'evidence': enriched_evidence,
    }
    log.debug('HYPOTHESIS input keys: %s', list(enriched_evidence.keys()))
    result = client.call_flash(HYPOTHESIS_PROMPT, payload)
    log.debug('HYPOTHESIS raw response: %s', {k: v for k, v in result.items() if k != 'evidence'})
    hyp = Hypothesis(
        finding_id=finding.finding_id,
        headline=result.get('headline', ''),
        explanation=json.dumps({
            'ds_explanation':       result.get('ds_explanation', ''),
            'business_explanation': result.get('business_explanation', ''),
            'suggested_fix':        result.get('suggested_fix', ''),
        }),
        evidence_refs=result.get('evidence_refs', []),
        confidence=result.get('confidence', 'low'),
    )
    log.info('HYPOTHESIS [%s] headline=%r confidence=%s',
             finding.finding_type, hyp.headline, hyp.confidence)
    return hyp


def run_critic(
    client: DeepSeekClient,
    finding: CandidateFinding,
    hypothesis: Hypothesis,
    enriched_evidence: dict[str, Any],
    grounding_advisory: dict[str, Any] | None = None,
) -> Critique:
    payload = {
        'finding_type': finding.finding_type,
        'finding_summary': finding.summary,
        'evidence': enriched_evidence,
        'hypothesis': {
            'headline':      hypothesis.headline,
            'explanation':   json.loads(hypothesis.explanation) if hypothesis.explanation else {},
            'evidence_refs': hypothesis.evidence_refs,
            'confidence':    hypothesis.confidence,
        },
    }
    if grounding_advisory is not None:
        payload['grounding_advisory'] = grounding_advisory
    log.debug('CRITIC input: headline=%r confidence=%s grounding_ok=%s',
              hypothesis.headline, hypothesis.confidence,
              grounding_advisory.get('grounding_ok') if grounding_advisory else 'n/a')
    result = client.call_pro(CRITIC_PROMPT, payload)
    log.debug('CRITIC raw response: %s', result)
    critique = Critique(
        finding_id=finding.finding_id,
        status=result.get('status', 'needs_review'),
        confidence=result.get('confidence', 'low'),
        notes=result.get('notes', ''),
        overclaim=bool(result.get('overclaim', False)),
        causal_external=bool(result.get('causal_external', False)),
    )
    log.info('CRITIC [%s] status=%s confidence=%s overclaim=%s causal_external=%s',
             finding.finding_type, critique.status, critique.confidence,
             critique.overclaim, critique.causal_external)
    return critique


# ── Async agent functions (used by graph.py for concurrent fan-out) ───────────

async def run_planner_async(client: DeepSeekClient, finding: CandidateFinding) -> list[str]:
    payload = {
        'finding_type': finding.finding_type,
        'score': finding.score,
        'summary': finding.summary,
        'evidence_keys_available': list(finding.evidence.keys()),
    }
    log.debug('PLANNER input: finding_type=%s score=%.2f', finding.finding_type, finding.score)
    result = await client.acall_flash(PLANNER_PROMPT, payload)
    valid = [t for t in result.get('tools', []) if t in _VALID_TOOLS]
    if not valid:
        log.warning('PLANNER returned no valid tools, falling back to defaults')
        valid = ['read_forecast_accuracy', 'read_recurring_drivers']
    log.info('PLANNER [%s] → %s | %s', finding.finding_type, valid, result.get('rationale', ''))
    return valid


async def run_hypothesis_async(
    client: DeepSeekClient,
    finding: CandidateFinding,
    enriched_evidence: dict[str, Any],
) -> Hypothesis:
    payload = {
        'finding_type': finding.finding_type,
        'summary': finding.summary,
        'score': finding.score,
        'evidence': enriched_evidence,
    }
    log.debug('HYPOTHESIS input keys: %s', list(enriched_evidence.keys()))
    result = await client.acall_flash(HYPOTHESIS_PROMPT, payload)
    hyp = Hypothesis(
        finding_id=finding.finding_id,
        headline=result.get('headline', ''),
        explanation=json.dumps({
            'ds_explanation':       result.get('ds_explanation', ''),
            'business_explanation': result.get('business_explanation', ''),
            'suggested_fix':        result.get('suggested_fix', ''),
        }),
        evidence_refs=result.get('evidence_refs', []),
        confidence=result.get('confidence', 'low'),
    )
    log.info('HYPOTHESIS [%s] headline=%r confidence=%s',
             finding.finding_type, hyp.headline, hyp.confidence)
    return hyp


async def run_critic_async(
    client: DeepSeekClient,
    finding: CandidateFinding,
    hypothesis: Hypothesis,
    enriched_evidence: dict[str, Any],
    grounding_advisory: dict[str, Any] | None = None,
) -> Critique:
    payload = {
        'finding_type': finding.finding_type,
        'finding_summary': finding.summary,
        'evidence': enriched_evidence,
        'hypothesis': {
            'headline':      hypothesis.headline,
            'explanation':   json.loads(hypothesis.explanation) if hypothesis.explanation else {},
            'evidence_refs': hypothesis.evidence_refs,
            'confidence':    hypothesis.confidence,
        },
    }
    if grounding_advisory is not None:
        payload['grounding_advisory'] = grounding_advisory
    log.debug('CRITIC input: headline=%r confidence=%s', hypothesis.headline, hypothesis.confidence)
    result = await client.acall_pro(CRITIC_PROMPT, payload)
    critique = Critique(
        finding_id=finding.finding_id,
        status=result.get('status', 'needs_review'),
        confidence=result.get('confidence', 'low'),
        notes=result.get('notes', ''),
        overclaim=bool(result.get('overclaim', False)),
        causal_external=bool(result.get('causal_external', False)),
    )
    log.info('CRITIC [%s] status=%s confidence=%s overclaim=%s causal_external=%s',
             finding.finding_type, critique.status, critique.confidence,
             critique.overclaim, critique.causal_external)
    return critique


async def run_business_synthesis(
    client: DeepSeekClient,
    accepted_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flash: VP-facing progress + phased plan."""
    payload = {
        'n_accepted_findings': len(accepted_findings),
        'findings': accepted_findings,
    }
    log.debug('BUSINESS_SYNTHESIS input: %d accepted findings', len(accepted_findings))
    result = await client.acall_flash(BUSINESS_SYNTHESIS_PROMPT, payload)
    log.info('BUSINESS_SYNTHESIS overall_confidence=%s headline=%s',
             result.get('overall_confidence'), result.get('headline', '')[:60])
    return result


async def run_technical_synthesis(
    client: DeepSeekClient,
    accepted_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flash: DS-facing bucketed levers grounded in evidence."""
    payload = {
        'n_accepted_findings': len(accepted_findings),
        'findings': accepted_findings,
    }
    log.debug('TECHNICAL_SYNTHESIS input: %d accepted findings', len(accepted_findings))
    result = await client.acall_flash(TECHNICAL_SYNTHESIS_PROMPT, payload)
    log.info('TECHNICAL_SYNTHESIS overall_confidence=%s levers=%d',
             result.get('overall_confidence'), len(result.get('levers', [])))
    return result
