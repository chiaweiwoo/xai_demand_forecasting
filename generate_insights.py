"""
Evidence-first XAI insights generation.

Reads from:  db/forecasting.db (evaluations, xai_results, external_signals, features, weekly_sales)
Writes to:   db/forecasting.db (insight_findings, insight_summary)

Requires DEEPSEEK_API_KEY in .env. Fails loudly if key is absent.
Run run_xai.py first to populate xai_results.
Safe to re-run — clears insight tables at start and regenerates all.

Next: uv run python data_quality.py
"""

import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from xai_forecast.db import (
    get_conn, insert_insight_finding, insert_insight_summary, load_insight_findings,
)
from xai_forecast.insights.llm_client import DeepSeekClient
from xai_forecast.insights.graph import run_insights_graph

DB_PATH = 'db/forecasting.db'


def main() -> None:
    print('Loading DB...')
    conn = get_conn(DB_PATH)

    # Verify XAI results exist
    n_xai = conn.execute('SELECT COUNT(*) FROM xai_results').fetchone()[0]
    if n_xai == 0:
        print('No XAI results found. Run: uv run python run_xai.py')
        conn.close()
        return

    n_bad = conn.execute(
        'SELECT COUNT(DISTINCT week_id) FROM evaluations WHERE is_bad_week=1'
    ).fetchone()[0]
    print(f'{n_xai:,} XAI rows across {n_bad} bad weeks')

    # Require LLM — fail loudly
    print('Connecting to DeepSeek...')
    client = DeepSeekClient()  # raises RuntimeError if key absent
    print(f'  Flash model: {client.flash_model}')
    print(f'  Critic model: {client.pro_model}')

    # Clean slate
    conn.execute('DELETE FROM insight_findings')
    conn.execute('DELETE FROM insight_summary')
    conn.commit()
    print('Cleared insight tables.')

    # Run the LangGraph pipeline
    print('\nRunning insights graph...')
    ledger_rows, summary = run_insights_graph(conn, client)

    # Persist findings ledger
    from datetime import datetime, UTC
    for row in ledger_rows:
        insert_insight_finding(conn, {
            'finding_id':   row.finding_id,
            'finding_type': row.finding_type,
            'status':       row.status,
            'confidence':   row.confidence,
            'evidence':     json.dumps(row.evidence),
            'hypothesis':   json.dumps(row.hypothesis) if row.hypothesis else None,
            'critic_notes': row.critic_notes,
            'created_at':   datetime.now(UTC).isoformat(),
        })

    # Persist summary
    ds      = summary.get('data_scientist', {})
    biz     = summary.get('business_leader', {})
    insert_insight_summary(conn, ds, biz, client.flash_model, client.pro_model)

    conn.close()

    # Summary
    n_accepted = sum(1 for r in ledger_rows if r.status == 'accepted')
    n_rejected = sum(1 for r in ledger_rows if r.status == 'rejected')
    n_review   = sum(1 for r in ledger_rows if r.status == 'needs_review')
    print(
        f'\nDone — {len(ledger_rows)} findings: '
        f'{n_accepted} accepted, {n_rejected} rejected, {n_review} needs_review'
    )
    print(f'Summary confidence: {summary.get("overall_confidence", "?")}')
    if ds.get('headline'):
        print(f'\nDS:  {ds["headline"]}')
    if biz.get('headline'):
        print(f'Biz: {biz["headline"]}')
    print('\nNext: uv run python data_quality.py')


if __name__ == '__main__':
    main()
