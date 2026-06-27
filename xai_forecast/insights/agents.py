"""
LLM steps for the insights pipeline.

Three prompt constants (run /prompt-audit before editing):
  HYPOTHESIS_PROMPT  -- Flash: interpret one evidence pack, write grounded hypothesis
  CRITIC_PROMPT      -- Pro:   reject overclaim, forbid causal external claims, set confidence
  SYNTHESIS_PROMPT   -- Flash: combine accepted findings into two-perspective summary

Each function is a pure LLM call: takes evidence/text, returns a dict.
"""

from __future__ import annotations

import json
from typing import Any

from .llm_client import DeepSeekClient
from .schemas import CandidateFinding, Hypothesis, Critique

# ── Prompt constants ──────────────────────────────────────────────────────────
# NOTE: These are *_PROMPT constants. Per project rules, run /prompt-audit
# before committing any edit to these strings.

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


# ── Agent step functions ──────────────────────────────────────────────────────

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
    result = client.call_flash(HYPOTHESIS_PROMPT, payload)
    return Hypothesis(
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
    result = client.call_pro(CRITIC_PROMPT, payload)
    return Critique(
        finding_id=finding.finding_id,
        status=result.get('status', 'needs_review'),
        confidence=result.get('confidence', 'low'),
        notes=result.get('notes', ''),
        overclaim=bool(result.get('overclaim', False)),
        causal_external=bool(result.get('causal_external', False)),
    )


def run_synthesis(
    client: DeepSeekClient,
    accepted_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flash: combine accepted findings into a two-perspective summary."""
    payload = {
        'n_accepted_findings': len(accepted_findings),
        'findings': accepted_findings,
    }
    return client.call_flash(SYNTHESIS_PROMPT, payload)
