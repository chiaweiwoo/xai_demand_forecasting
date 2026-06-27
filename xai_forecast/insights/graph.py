"""
LangGraph orchestrator for the insights pipeline.

Graph shape:
  detect_candidates → fan-out per finding (hypothesis → critic) → fan-in → synthesize

conn and client are captured in node-function closures — they never enter the
LangGraph state dict, which keeps the state serialisation-safe and avoids
KeyError when LangGraph passes only the node-output delta to edge routers.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from operator import add
from typing import Annotated, Any, TypedDict

log = logging.getLogger(__name__)

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents import run_planner, run_hypothesis, run_critic, run_synthesis
from .detectors import run_all_detectors
from .llm_client import DeepSeekClient
from .schemas import CandidateFinding, LedgerRow
from .tools import (
    read_forecast_accuracy,
    read_bad_weeks,
    read_xai_findings,
    read_demand_trajectory,
    read_external_signals,
    read_model_metadata,
    read_recurring_drivers,
)


def _enrich_evidence(
    conn: sqlite3.Connection,
    finding: CandidateFinding,
    tools_to_run: list[str],
) -> dict[str, Any]:
    """Call the planner-chosen read-tools and merge results into the evidence dict."""
    enriched = dict(finding.evidence)

    for tool_name in tools_to_run:
        try:
            if tool_name == 'read_forecast_accuracy':
                enriched['forecast_accuracy'] = read_forecast_accuracy(conn)

            elif tool_name == 'read_recurring_drivers':
                enriched['recurring_drivers'] = read_recurring_drivers(conn)[:10]

            elif tool_name == 'read_bad_weeks':
                enriched['bad_weeks'] = read_bad_weeks(conn)

            elif tool_name == 'read_model_metadata':
                enriched['model_metadata'] = read_model_metadata(conn)

            elif tool_name == 'read_xai_findings':
                bad_weeks = read_bad_weeks(conn)
                if bad_weeks:
                    worst = sorted(bad_weeks, key=lambda x: x.get('avg_mape') or 0, reverse=True)[0]
                    enriched['sample_xai'] = read_xai_findings(
                        conn, week_id=worst['week_id'], xai_type='shap'
                    )[:5]

            elif tool_name == 'read_demand_trajectory':
                examples = finding.evidence.get('top_examples', [])[:3]
                trajectories = []
                for ex in examples:
                    traj = read_demand_trajectory(conn, ex['item_id'], ex['week_id'])
                    trajectories.append(traj)
                if trajectories:
                    enriched['demand_trajectories'] = trajectories

            elif tool_name == 'read_external_signals':
                notable = finding.evidence.get('notable_weeks', [])[:5]
                ext_data = []
                for n in notable:
                    sig = read_external_signals(conn, n['week'])
                    if sig:
                        ext_data.append(sig)
                if ext_data:
                    enriched['external_signals_detail'] = ext_data

        except Exception as exc:
            enriched[f'tool_error_{tool_name}'] = str(exc)

    return enriched


def _flatten_keys(d: dict, prefix: str = '') -> list[str]:
    """Return all dot-path keys in a nested dict, e.g. 'forecast_accuracy.n_bad_weeks'."""
    keys = []
    for k, v in d.items():
        full = f'{prefix}.{k}' if prefix else k
        keys.append(full)
        if isinstance(v, dict):
            keys.extend(_flatten_keys(v, full))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            keys.extend(_flatten_keys(v[0], f'{full}.0'))
    return keys


# ── Graph builder (factory — closes over conn + client) ───────────────────────

class _State(TypedDict):
    candidates:  list
    ledger_rows: Annotated[list, add]  # fan-in reducer: each review_finding appends its row
    summary:     dict


def _build_graph(conn: sqlite3.Connection, client: DeepSeekClient):
    """Build and compile the StateGraph, capturing conn/client in closures."""

    def detect_candidates(state: dict) -> dict:
        log.info('Running detectors...')
        candidates = run_all_detectors(conn)
        log.info('%d candidate findings', len(candidates))
        for c in candidates:
            log.debug('  candidate: %s score=%.2f — %s', c.finding_type, c.score, c.summary)
        return {'candidates': candidates}

    def route_findings(state: dict):
        """Fan-out: one Send per candidate finding."""
        candidates = state.get('candidates', [])
        if not candidates:
            log.info('No candidates — jumping to synthesize')
            return 'synthesize'
        log.info('Routing %d findings for review', len(candidates))
        return [
            Send('review_finding', {'finding': c})
            for c in candidates
        ]

    def review_finding(state: dict) -> dict:
        """Per-finding node: plan → enrich → hypothesis → grounding check → critic → LedgerRow."""
        finding: CandidateFinding = state['finding']
        ft = finding.finding_type

        log.info('[%s] planning evidence gathering (Flash)...', ft)
        tools_to_run = run_planner(client, finding)
        log.info('[%s] planner chose: %s', ft, tools_to_run)

        log.info('[%s] enriching evidence with %d tools...', ft, len(tools_to_run))
        enriched = _enrich_evidence(conn, finding, tools_to_run)
        log.debug('[%s] enriched evidence keys: %s', ft, list(enriched.keys()))

        log.info('[%s] running hypothesis (Flash)...', ft)
        hypothesis = run_hypothesis(client, finding, enriched)
        log.info('[%s] hypothesis: headline=%r confidence=%s', ft, hypothesis.headline, hypothesis.confidence)
        log.debug('[%s] hypothesis evidence_refs: %s', ft, hypothesis.evidence_refs)

        # Deterministic grounding check: verify evidence_refs point to real evidence keys.
        if hypothesis.evidence_refs:
            evidence_keys = set(_flatten_keys(enriched))
            missing = [r for r in hypothesis.evidence_refs if r not in evidence_keys]
            if missing:
                log.warning('[%s] GROUNDING FAIL — refs not in evidence: %s', ft, missing)
                hypothesis.confidence = 'low'
            else:
                log.debug('[%s] grounding OK — all %d refs found', ft, len(hypothesis.evidence_refs))

        # Skip Pro critic for low-confidence hypotheses.
        if hypothesis.confidence == 'low':
            log.info('[%s] skipping critic (confidence=low → needs_review)', ft)
            from xai_forecast.insights.schemas import Critique
            critique = Critique(
                finding_id=finding.finding_id,
                status='needs_review',
                confidence='low',
                notes='Skipped Pro critic: hypothesis confidence was low (thin evidence or grounding failure).',
                overclaim=False,
                causal_external=False,
            )
        else:
            log.info('[%s] running critic (Pro)...', ft)
            critique = run_critic(client, finding, hypothesis, enriched)
            log.info('[%s] critic: status=%s confidence=%s overclaim=%s causal_external=%s',
                     ft, critique.status, critique.confidence, critique.overclaim, critique.causal_external)
            log.debug('[%s] critic notes: %s', ft, critique.notes)

        log.info('[%s] RESULT: status=%s confidence=%s', ft, critique.status, critique.confidence)

        row = LedgerRow(
            finding_id=finding.finding_id,
            finding_type=finding.finding_type,
            status=critique.status,
            confidence=critique.confidence,
            evidence=enriched,
            hypothesis={
                'headline':      hypothesis.headline,
                'explanation':   json.loads(hypothesis.explanation) if hypothesis.explanation else {},
                'evidence_refs': hypothesis.evidence_refs,
            } if critique.status != 'rejected' else None,
            critic_notes=critique.notes,
        )
        return {'ledger_rows': [row]}

    def synthesize(state: dict) -> dict:
        """Fan-in: combine accepted findings into the two-perspective summary."""
        rows = state.get('ledger_rows', [])

        accepted = [
            {
                'finding_id':   r.finding_id,
                'finding_type': r.finding_type,
                'confidence':   r.confidence,
                'hypothesis':   r.hypothesis,
                'critic_notes': r.critic_notes,
            }
            for r in rows
            if r.status == 'accepted'
        ]

        log.info('Synthesis: %d accepted / %d total (Flash)...', len(accepted), len(rows))

        if not accepted:
            log.warning('No accepted findings — returning inconclusive summary')
            summary = {
                'data_scientist': {
                    'headline': 'Insufficient evidence for confident findings',
                    'summary':  'All candidate findings were rejected or flagged for review.',
                    'top_issues': [],
                    'recommended_actions': ['Collect more backtest weeks before re-running insights'],
                },
                'business_leader': {
                    'headline': 'Analysis inconclusive — more data needed',
                    'summary':  'The automated review could not confirm specific failure patterns.',
                    'risk_direction': 'mixed',
                    'limitations': ['Insufficient data for confident conclusions'],
                    'improvement_plan': 'Extend the backtest period and re-run the insights analysis.',
                },
                'overall_confidence': 'low',
            }
        else:
            summary = run_synthesis(client, accepted)
            log.info('Synthesis complete: overall_confidence=%s', summary.get('overall_confidence'))
            log.debug('DS headline: %s', summary.get('data_scientist', {}).get('headline'))
            log.debug('Biz headline: %s', summary.get('business_leader', {}).get('headline'))

        return {'summary': summary}

    builder = StateGraph(_State)
    builder.add_node('detect_candidates', detect_candidates)
    builder.add_node('review_finding',    review_finding)
    builder.add_node('synthesize',        synthesize)

    builder.add_edge(START, 'detect_candidates')
    builder.add_conditional_edges('detect_candidates', route_findings)
    builder.add_edge('review_finding', 'synthesize')
    builder.add_edge('synthesize', END)

    return builder.compile(name='xai-insights-graph')


def run_insights_graph(
    conn: sqlite3.Connection,
    client: DeepSeekClient,
) -> tuple[list[LedgerRow], dict[str, Any]]:
    """
    Entry point. Returns (ledger_rows, summary_dict).
    ledger_rows includes all findings (accepted + rejected + needs_review).
    summary_dict is the two-perspective synthesis.
    """
    graph = _build_graph(conn, client)
    result = graph.invoke(
        {
            'candidates':  [],
            'ledger_rows': [],
            'summary':     {},
        },
    )
    return result.get('ledger_rows', []), result.get('summary', {})
