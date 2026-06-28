"""
Evidence-first XAI insights generation.

Reads from:  db/forecasting.db (evaluations, xai_results, external_signals, features, weekly_sales)
Writes to:   db/forecasting.db (insight_findings, insight_summary)
Logs to:     logs/insights.log  (overwritten each run — full agent trace for debugging)

Requires DEEPSEEK_API_KEY in .env. Fails loudly if key is absent.
Run run_xai.py first to populate xai_results.
Safe to re-run — clears insight tables at start and regenerates all.

Next: uv run python data_quality.py
"""

import json
import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _setup_logging() -> logging.Logger:
    """Configure root logger: INFO to console, DEBUG to logs/insights.log."""
    Path('logs').mkdir(exist_ok=True)
    log_path = 'logs/insights.log'

    fmt = '%(asctime)s  %(levelname)-8s  %(name)s  %(message)s'
    datefmt = '%H:%M:%S'

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Console — INFO only (same as the current print output)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)

    # File — DEBUG (full agent trace)
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)

    return logging.getLogger('insights')


logger = _setup_logging()

from xai_forecast.db import (
    get_conn, insert_insight_finding, insert_insight_summary, load_insight_findings,
)
from xai_forecast.insights.llm_client import DeepSeekClient
from xai_forecast.insights.graph import run_insights_graph

DB_PATH = 'db/forecasting.db'


def main() -> None:
    logger.info('Loading DB: %s', DB_PATH)
    conn = get_conn(DB_PATH)

    n_xai = conn.execute('SELECT COUNT(*) FROM xai_results').fetchone()[0]
    if n_xai == 0:
        logger.error('No XAI results found. Run: uv run python run_xai.py')
        conn.close()
        return

    n_bad = conn.execute(
        'SELECT COUNT(DISTINCT week_id) FROM evaluations WHERE is_bad_week=1'
    ).fetchone()[0]
    logger.info('%s XAI rows across %s bad weeks', f'{n_xai:,}', n_bad)

    logger.info('Connecting to DeepSeek...')
    client = DeepSeekClient()
    logger.info('  Flash model: %s', client.flash_model)
    logger.info('  Critic model: %s', client.pro_model)

    conn.execute('DELETE FROM insight_findings')
    conn.execute('DELETE FROM insight_summary')
    conn.commit()
    logger.info('Cleared insight tables.')

    logger.info('Running insights graph...')
    ledger_rows, summary = run_insights_graph(conn, client)

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
        logger.debug('Persisted finding: %s | status=%s confidence=%s',
                     row.finding_id, row.status, row.confidence)

    ds  = summary.get('data_scientist', {})
    biz = summary.get('business_leader', {})
    # overall_confidence is a top-level synthesis key — fold it into both perspective
    # dicts so it survives persistence (insert_insight_summary stores only these two).
    overall_conf = summary.get('overall_confidence', 'medium')
    ds.setdefault('overall_confidence', overall_conf)
    biz.setdefault('overall_confidence', overall_conf)
    insert_insight_summary(conn, ds, biz, client.flash_model, client.pro_model)
    conn.close()

    n_accepted = sum(1 for r in ledger_rows if r.status == 'accepted')
    n_rejected = sum(1 for r in ledger_rows if r.status == 'rejected')
    n_review   = sum(1 for r in ledger_rows if r.status == 'needs_review')

    logger.info('Done — %d findings: %d accepted, %d rejected, %d needs_review',
                len(ledger_rows), n_accepted, n_rejected, n_review)
    logger.info('Summary confidence: %s', summary.get('overall_confidence', '?'))
    if ds.get('headline'):
        logger.info('DS:  %s', ds['headline'])
    if biz.get('headline'):
        logger.info('Biz: %s', biz['headline'])
    logger.info('Log written to: logs/insights.log')
    logger.info('Next: uv run python data_quality.py')


if __name__ == '__main__':
    main()
