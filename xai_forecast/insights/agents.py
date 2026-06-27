"""
LLM steps for the insights pipeline.

Four prompt constants (run /prompt-audit before editing):
  PLANNER_PROMPT     -- Flash: decide which read-tools to call for this finding
  HYPOTHESIS_PROMPT  -- Flash: interpret one evidence pack, write grounded hypothesis
  CRITIC_PROMPT      -- Pro:   reject overclaim, forbid causal external claims, set confidence
  SYNTHESIS_PROMPT   -- Flash: combine accepted findings into two-perspective summary

Each function is a pure LLM call: takes evidence/text, returns a dict.
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
- If evidence contains external signal conditions: state them as CORRELATION only.
  Never say "caused by" or "due to" for external signals — say "coincided with" or "during a period of".
- For over_forecast_bias or dominant_driver: state the risk direction explicitly
  (e.g. "over-ordering risk", "excess inventory bias").
- For demand_cliff: quote the actual lag_1 and actual sales numbers from the top example.
- If evidence is absent or contradictory with no clear pattern, set confidence=low AND
  note the gap in ds_explanation — do not fabricate a direction.
- Before writing final JSON, verify: (1) every number came from the evidence JSON,
  (2) no causal claim for external signals, (3) suggested_fix is actionable.
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
- Downgrade confidence if: evidence rows < 3, pattern not consistent across weeks,
  or the finding could have an equally plausible alternative explanation.
- Accept if: all claims traceable to the evidence pack, correlation-only for external,
  and the finding is specific enough to act on.
- causal_external = true if the hypothesis says external signals CAUSED the error.
  If causal_external is true, status MUST be "rejected".
- overclaim = true if hypothesis asserts facts not present in the evidence.
- Before writing your JSON, re-read notes and confirm status is consistent:
  if notes identifies overclaim, overclaim must be true and status must not be "accepted".
- Respond in English only."""

SYNTHESIS_PROMPT = """\
You are a senior AI/ML governance analyst writing an executive summary of accepted XAI findings
for two audiences: a data scientist (what to fix) and a business leader (what it means).

Rules:
- Use ONLY the accepted findings provided. Do NOT invent patterns not present in the findings.
- Return valid JSON only — no markdown, no code fences:
  {"data_scientist": {
     "headline": "<15-20 word summary>",
     "summary": "<3-4 sentences: root causes, model structural issues, concrete fixes>",
     "top_issues": ["<issue 1>", "<issue 2>", "<issue 3>"],
     "recommended_actions": ["<action 1>", "<action 2>", "<action 3>"]
   },
   "business_leader": {
     "headline": "<15-20 word summary suitable for a board update>",
     "summary": "<3-4 sentences: what failed, risk direction, limitations, improvement plan>",
     "risk_direction": "<over-stock|under-stock|mixed>",
     "limitations": ["<limitation 1>", "<limitation 2>"],
     "improvement_plan": "<2-3 sentences: what will be done and expected impact>"
   },
   "overall_confidence": "<high|medium|low>"}
- If n_accepted_findings is 0, return JSON with headline "No accepted findings cleared the quality gate",
  summary "No patterns were confirmed by the critic.", top_issues/recommended_actions/limitations as [],
  improvement_plan "Rerun with more backtest weeks.", risk_direction "mixed", overall_confidence "low".
  Do NOT synthesize from rejected findings.
- top_issues and recommended_actions: include up to 3 items — only what the evidence supports.
  Do not pad to 3 if fewer are warranted.
- For the business summary: state the risk direction (over-forecast = over-ordering risk),
  avoid all model jargon (SHAP, log-margin, LightGBM), and focus on business impact.
- For the data scientist summary: be specific — name the features, metrics, thresholds.
- overall_confidence = high if >= 3 high-confidence findings; medium if 1-2; low otherwise.
- Respond in English only.
- Before writing final JSON, verify: (1) risk_direction matches the over/under-forecast evidence,
  (2) DS summary mentions specific features, (3) business summary has no model jargon."""


_VALID_TOOLS = {
    'read_forecast_accuracy',
    'read_bad_weeks',
    'read_xai_findings',
    'read_demand_trajectory',
    'read_external_signals',
    'read_model_metadata',
    'read_recurring_drivers',
}


# ── Agent step functions ──────────────────────────────────────────────────────

def run_planner(
    client: DeepSeekClient,
    finding: CandidateFinding,
) -> list[str]:
    """Flash: decide which read-tools to call for this finding. Returns a validated tool list."""
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
    chosen = result.get('tools', [])
    rationale = result.get('rationale', '')
    valid = [t for t in chosen if t in _VALID_TOOLS]
    if not valid:
        log.warning('PLANNER returned no valid tools, falling back to defaults')
        valid = ['read_forecast_accuracy', 'read_recurring_drivers']
    log.info('PLANNER [%s] → %s | %s', finding.finding_type, valid, rationale)
    return valid


def run_hypothesis(
    client: DeepSeekClient,
    finding: CandidateFinding,
    enriched_evidence: dict[str, Any],
) -> Hypothesis:
    """Flash: turn an evidence pack into a grounded hypothesis."""
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
    log.info('HYPOTHESIS [%s] headline=%r confidence=%s refs=%s',
             finding.finding_type, hyp.headline, hyp.confidence, hyp.evidence_refs)
    return hyp


def run_critic(
    client: DeepSeekClient,
    finding: CandidateFinding,
    hypothesis: Hypothesis,
    enriched_evidence: dict[str, Any],
) -> Critique:
    """Pro: reject overclaim, enforce correlation-only for external signals."""
    payload = {
        'finding_type': finding.finding_type,
        'finding_summary': finding.summary,
        'evidence': enriched_evidence,
        'hypothesis': {
            'headline':     hypothesis.headline,
            'explanation':  json.loads(hypothesis.explanation) if hypothesis.explanation else {},
            'evidence_refs': hypothesis.evidence_refs,
            'confidence':    hypothesis.confidence,
        },
    }
    log.debug('CRITIC input: headline=%r confidence=%s', hypothesis.headline, hypothesis.confidence)
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
    log.info('CRITIC [%s] status=%s confidence=%s overclaim=%s causal_external=%s notes=%r',
             finding.finding_type, critique.status, critique.confidence,
             critique.overclaim, critique.causal_external, critique.notes[:120])
    return critique


def run_synthesis(
    client: DeepSeekClient,
    accepted_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flash: combine accepted findings into a two-perspective summary."""
    payload = {
        'n_accepted_findings': len(accepted_findings),
        'findings': accepted_findings,
    }
    log.debug('SYNTHESIS input: %d accepted findings', len(accepted_findings))
    result = client.call_flash(SYNTHESIS_PROMPT, payload)
    log.info('SYNTHESIS overall_confidence=%s', result.get('overall_confidence'))
    log.debug('SYNTHESIS DS headline: %s', result.get('data_scientist', {}).get('headline'))
    log.debug('SYNTHESIS Biz headline: %s', result.get('business_leader', {}).get('headline'))
    return result
