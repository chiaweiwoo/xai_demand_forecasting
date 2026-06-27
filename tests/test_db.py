"""Group D — DB / storage tests."""

import json

import numpy as np
import pandas as pd
import pytest

from xai_forecast.db import (
    get_conn,
    insert_forecasts, insert_evaluations, insert_xai,
    insert_insight_finding, insert_insight_summary,
    load_insight_summary, load_insight_findings,
    load_features_week,
    week_summary,
)
from xai_forecast.features import FEATURE_COLS


# ── INSERT OR REPLACE idempotency ─────────────────────────────────────────────

def test_insert_forecasts_idempotent(db_conn):
    """Second INSERT with same PK should overwrite, not duplicate."""
    row = {'week_id': '2015-01-01', 'item_id': 'CA_1_001_TX_1', 'h1': 5.0, 'trained_at': 'now'}
    insert_forecasts(db_conn, [row])
    row['h1'] = 9.0
    insert_forecasts(db_conn, [row])

    count = db_conn.execute("SELECT COUNT(*) FROM forecasts WHERE week_id='2015-01-01'").fetchone()[0]
    assert count == 1
    h1 = db_conn.execute("SELECT h1 FROM forecasts WHERE week_id='2015-01-01'").fetchone()[0]
    assert h1 == pytest.approx(9.0)


def test_insert_evaluations_idempotent(db_conn):
    row = {'week_id': '2015-01-01', 'item_id': 'A', 'h1_mape': 10.0,
           'h1_mae': 1.0, 'is_bad_week': 0, 'mape_zscore': 0.5}
    insert_evaluations(db_conn, [row])
    row['h1_mape'] = 99.0
    insert_evaluations(db_conn, [row])

    count = db_conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    assert count == 1
    mape = db_conn.execute("SELECT h1_mape FROM evaluations").fetchone()[0]
    assert mape == pytest.approx(99.0)


def test_insert_xai_idempotent(db_conn):
    row = {'week_id': '2015-01-01', 'item_id': 'A', 'xai_type': 'shap',
           'payload': json.dumps({'x': 1})}
    insert_xai(db_conn, [row])
    row['payload'] = json.dumps({'x': 2})
    insert_xai(db_conn, [row])

    count = db_conn.execute("SELECT COUNT(*) FROM xai_results").fetchone()[0]
    assert count == 1
    payload = json.loads(db_conn.execute("SELECT payload FROM xai_results").fetchone()[0])
    assert payload['x'] == 2


# ── Read-back round-trip ──────────────────────────────────────────────────────

def test_insert_forecasts_readback(db_conn):
    insert_forecasts(db_conn, [
        {'week_id': '2015-03-01', 'item_id': 'CA_1_042_TX_1', 'h1': 7.25, 'trained_at': 't'},
    ])
    row = db_conn.execute("SELECT * FROM forecasts WHERE week_id='2015-03-01'").fetchone()
    assert row['item_id'] == 'CA_1_042_TX_1'
    assert row['h1'] == pytest.approx(7.25)


def test_insert_evaluations_readback(db_conn):
    insert_evaluations(db_conn, [
        {'week_id': '2015-04-01', 'item_id': 'X', 'h1_mape': 33.3,
         'h1_mae': 2.5, 'is_bad_week': 1, 'mape_zscore': 2.1},
    ])
    row = db_conn.execute("SELECT * FROM evaluations WHERE week_id='2015-04-01'").fetchone()
    assert row['is_bad_week'] == 1
    assert row['mape_zscore'] == pytest.approx(2.1)


# ── Clean-slate DELETE ────────────────────────────────────────────────────────

def test_clean_slate_delete(db_conn):
    insert_forecasts(db_conn, [
        {'week_id': '2015-01-01', 'item_id': 'A', 'h1': 1.0, 'trained_at': 'x'},
        {'week_id': '2015-01-08', 'item_id': 'A', 'h1': 2.0, 'trained_at': 'x'},
    ])
    db_conn.executescript('DELETE FROM forecasts;')
    db_conn.commit()
    count = db_conn.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0]
    assert count == 0


# ── load_features_week: one row per SKU ──────────────────────────────────────

def test_load_features_week_one_row_per_sku(db_conn):
    """Writing 3 SKUs for the same week then reading back should return exactly 3 rows."""
    rows = []
    for uid in ['CA_1_001_TX_1', 'CA_1_002_TX_1', 'CA_1_003_TX_1']:
        row = {'unique_id': uid, 'week': '2015-06-01', 'y': 1.0}
        row.update({c: 0.0 for c in FEATURE_COLS})
        rows.append(row)

    pd.DataFrame(rows).to_sql('features', db_conn, if_exists='append', index=False)
    db_conn.commit()

    result = load_features_week(db_conn, '2015-06-01')
    assert len(result) == 3
    assert set(result['unique_id'].tolist()) == {'CA_1_001_TX_1', 'CA_1_002_TX_1', 'CA_1_003_TX_1'}


# ── week_summary only counts existing evaluations ────────────────────────────

def test_week_summary_counts(db_conn):
    insert_evaluations(db_conn, [
        {'week_id': '2015-01-01', 'item_id': 'A', 'h1_mape': 10.0, 'h1_mae': 1.0,
         'is_bad_week': 1, 'mape_zscore': 2.0},
        {'week_id': '2015-01-01', 'item_id': 'B', 'h1_mape': 5.0, 'h1_mae': 0.5,
         'is_bad_week': 0, 'mape_zscore': 0.3},
        {'week_id': '2015-01-08', 'item_id': 'A', 'h1_mape': 20.0, 'h1_mae': 2.0,
         'is_bad_week': 0, 'mape_zscore': 0.8},
    ])
    summary = week_summary(db_conn)
    assert len(summary) == 2
    w1 = summary[summary['week_id'] == '2015-01-01'].iloc[0]
    assert w1['n_items'] == 2
    assert w1['avg_mape'] == pytest.approx(7.5)
    assert w1['n_bad_items'] == 1


# ── Insight CRUD ─────────────────────────────────────────────────────────────

import json as _json
from datetime import datetime


def _finding_dict(**kwargs):
    defaults = dict(
        finding_id='test-001', finding_type='over_forecast_bias',
        status='accepted', confidence='high',
        evidence=_json.dumps({'pct_over': 100.0}),
        hypothesis=_json.dumps({'headline': 'All over-forecasts'}),
        critic_notes='No overclaim detected.',
        created_at=datetime.utcnow().isoformat(),
    )
    defaults.update(kwargs)
    return defaults


def test_insert_insight_finding_readback(db_conn):
    insert_insight_finding(db_conn, _finding_dict())
    findings = load_insight_findings(db_conn)
    assert len(findings) == 1
    assert findings[0]['finding_id'] == 'test-001'
    assert findings[0]['status'] == 'accepted'
    assert findings[0]['evidence']['pct_over'] == 100.0


def test_insert_insight_finding_idempotent(db_conn):
    """Second INSERT with same finding_id must overwrite, not duplicate."""
    insert_insight_finding(db_conn, _finding_dict(
        finding_id='test-002', finding_type='demand_cliff',
        status='pending', confidence='low',
        evidence=_json.dumps({'n': 3}), hypothesis=None, critic_notes=None,
    ))
    insert_insight_finding(db_conn, _finding_dict(
        finding_id='test-002', finding_type='demand_cliff',
        status='accepted', confidence='high',
        evidence=_json.dumps({'n': 5}),
        hypothesis=_json.dumps({'headline': 'Updated'}), critic_notes='ok',
    ))
    findings = load_insight_findings(db_conn)
    assert len(findings) == 1
    assert findings[0]['status'] == 'accepted'


def test_insert_insight_summary_readback(db_conn):
    insert_insight_summary(
        db_conn,
        data_scientist={'text': 'Fix rolling features.'},
        business_leader={'text': 'Model over-forecasts by 30%.'},
        model_flash='deepseek-v4-flash',
        model_critic='deepseek-v4-pro',
    )
    summary = load_insight_summary(db_conn)
    assert summary is not None
    assert 'data_scientist' in summary
    assert summary['data_scientist']['text'] == 'Fix rolling features.'
    assert summary['business_leader']['text'] == 'Model over-forecasts by 30%.'


def test_load_insight_summary_missing_returns_none(db_conn):
    result = load_insight_summary(db_conn)
    assert result is None
