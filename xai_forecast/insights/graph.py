"""
LangGraph orchestrator for the insights pipeline.

Graph shape:
  detect_candidates → fan-out per finding (hypothesis → critic) → fan-in → synthesize

Each finding node:
  1. Planner chooses which read-tools to call for this finding type
  2. Calls the read-tools to enrich the evidence pack
  3. Runs hypothesis (Flash)
  4. Runs critic (Pro) — rejects overclaim, enforces correlation-only for external signals
  5. Returns a LedgerRow

All accepted rows are fanned in and fed to a single synthesis call (Flash) that
produces the two-perspective summary.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, UTC
from functools import lru_cache
from operator import add
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents import run_hypothesis, run_critic, run_synthesis
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


# ── Shared state ──────────────────────────────────────────────────────────────

class RunState(dict):
    """TypedDict-compatible; keys defined here for documentation."""
    # conn: sqlite3.Connection           (set by caller — not serialisable)
    # client: DeepSeekClient             (set by caller)
    # candidates: list[CandidateFinding]
    # ledger_rows: list[LedgerRow]       (accumulated via Annotated[..., add])
    # summary: dict[str, Any]


class FindingTaskState(dict):
    """Per-finding fan-out state."""
    # conn: sqlite3.Connection
    # client: DeepSeekClient
    # finding: CandidateFinding


# ── Planner: choose tools per finding type ────────────────────────────────────

_TOOL_PLAN: dict[str, list[str]] = {
    'over_forecast_bias':       ['read_forecast_accuracy', 'read_recurring_drivers'],
    'dominant_driver':          ['read_recurring_drivers', 'read_forecast_accuracy', 'read_model_metadata'],
    'demand_cliff':             ['read_demand_trajectory', 'read_xai_findings', 'read_bad_weeks'],
    'external_coincidence':     ['read_bad_weeks', 'read_external_signals', 'read_xai_findings'],
    'counterfactual_material':  ['read_xai_findings', 'read_bad_weeks'],
    'contrastive_gap':          ['read_xai_findings', 'read_bad_weeks'],
}


def _enrich_evidence(
    conn: sqlite3.Connection,
    finding: CandidateFinding,
) -> dict[str, Any]:
    """Call the planned read-tools for this finding type and merge into evidence."""
    tools_to_run = _TOOL_PLAN.get(finding.finding_type, ['read_forecast_accuracy'])
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
                # Sample: top 5 SHAP payloads from the worst bad week
                bad_weeks = read_bad_weeks(conn)
                if bad_weeks:
                    worst = sorted(bad_weeks, key=lambda x: x.get('avg_mape') or 0, reverse=True)[0]
                    enriched['sample_xai'] = read_xai_findings(
                        conn, week_id=worst['week_id'], xai_type='shap'
                    )[:5]

            elif tool_name == 'read_demand_trajectory':
                # Pull trajectories for the top cliff examples in the finding
                examples = finding.evidence.get('top_examples', [])[:3]
                trajectories = []
                for ex in examples:
                    traj = read_demand_trajectory(conn, ex['item_id'], ex['week_id'])
                    trajectories.append(traj)
                if trajectories:
                    enriched['demand_trajectories'] = trajectories

            elif tool_name == 'read_external_signals':
                # Pull external signals for each notable week
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


# ── Graph nodes ───────────────────────────────────────────────────────────────

def _detect_candidates(state: RunState) -> dict:
    conn = state['conn']
    print('  Running detectors...')
    candidates = run_all_detectors(conn)
    print(f'  {len(candidates)} candidate findings')
    return {'candidates': candidates}


def _route_findings(state: RunState):
    """Fan-out: one Send per candidate finding."""
    candidates = state.get('candidates', [])
    if not candidates:
        return 'synthesize'
    return [
        Send(
            'review_finding',
            {
                'conn':    state['conn'],
                'client':  state['client'],
                'finding': c,
            },
        )
        for c in candidates
    ]


def _review_finding(state: FindingTaskState) -> dict:
    """Per-finding node: enrich → hypothesis → critic → LedgerRow."""
    conn    = state['conn']
    client  = state['client']
    finding = state['finding']

    print(f'  [{finding.finding_type}] enriching evidence...')
    enriched = _enrich_evidence(conn, finding)

    print(f'  [{finding.finding_type}] running hypothesis (Flash)...')
    hypothesis = run_hypothesis(client, finding, enriched)

    print(f'  [{finding.finding_type}] running critic (Pro)...')
    critique = run_critic(client, finding, hypothesis, enriched)

    print(f'  [{finding.finding_type}] status={critique.status} confidence={critique.confidence}')

    row = LedgerRow(
        finding_id=finding.finding_id,
        finding_type=finding.finding_type,
        status=critique.status,
        confidence=critique.confidence,
        evidence=enriched,
        hypothesis={
            'headline':     hypothesis.headline,
            'explanation':  json.loads(hypothesis.explanation) if hypothesis.explanation else {},
            'evidence_refs': hypothesis.evidence_refs,
        } if critique.status != 'rejected' else None,
        critic_notes=critique.notes,
    )
    return {'ledger_rows': [row]}


def _synthesize(state: RunState) -> dict:
    """Fan-in: combine accepted findings into the two-perspective summary."""
    client = state['client']
    rows   = state.get('ledger_rows', [])

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

    print(f'  Synthesis: {len(accepted)} accepted / {len(rows)} total findings (Flash)...')

    if not accepted:
        summary = {
            'data_scientist': {
                'headline': 'Insufficient evidence for confident findings',
                'summary':  'All candidate findings were rejected or flagged for review. Re-run after more backtest data is available.',
                'top_issues': [],
                'recommended_actions': ['Collect more backtest weeks before re-running insights'],
            },
            'business_leader': {
                'headline': 'Analysis inconclusive — more data needed',
                'summary':  'The automated review could not confirm specific failure patterns with sufficient confidence.',
                'risk_direction': 'mixed',
                'limitations': ['Insufficient data for confident conclusions'],
                'improvement_plan': 'Extend the backtest period and re-run the insights analysis.',
            },
            'overall_confidence': 'low',
        }
    else:
        summary = run_synthesis(client, accepted)

    return {'summary': summary}


# ── Graph builder ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_graph():
    builder = StateGraph(dict)

    builder.add_node('detect_candidates', _detect_candidates)
    builder.add_node('review_finding',    _review_finding)
    builder.add_node('synthesize',        _synthesize)

    builder.add_edge(START, 'detect_candidates')
    builder.add_conditional_edges('detect_candidates', _route_findings)
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
    graph = _build_graph()
    result = graph.invoke(
        {
            'conn':         conn,
            'client':       client,
            'candidates':   [],
            'ledger_rows':  [],
            'summary':      {},
        },
        config={'max_concurrency': 3},
    )
    return result.get('ledger_rows', []), result.get('summary', {})
