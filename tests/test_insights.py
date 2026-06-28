"""
Tests for the insights module.

Group E: detectors (deterministic), tools (read-only), LLM fallback, schemas.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from xai_forecast.db import get_conn
from xai_forecast.insights.schemas import CandidateFinding, EvidencePack, LedgerRow
from xai_forecast.insights.detectors import (
    detect_over_forecast_bias,
    detect_dominant_driver,
    detect_demand_cliff,
    detect_external_coincidence,
    detect_counterfactual_material,
    detect_contrastive_gap,
    run_all_detectors,
)
from xai_forecast.insights.tools import (
    read_forecast_accuracy,
    read_bad_weeks,
    read_xai_findings,
    read_external_signals,
    read_recurring_drivers,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_shap_payload(direction='over', lag1_val=50.0, actual=5.0, pred=48.0, feature='lag_1'):
    return json.dumps({
        'prediction': pred,
        'actual': actual,
        'base_value_log': 1.0,
        'error_pct': abs(pred - actual) / actual * 100 if actual > 0 else None,
        'signed_error': (pred - actual) / actual * 100 if actual > 0 else None,
        'direction': direction,
        'other_features_shap': 0.01,
        'top_features': [
            {'feature': feature, 'shap_value': 0.8, 'feature_value': lag1_val},
            {'feature': 'rolling_4_mean', 'shap_value': 0.4, 'feature_value': 45.0},
        ],
    })


def _make_cf_payload(pred=48.0, actual=5.0, snap_was_active=True, snap_delta=-5.0):
    return json.dumps({
        'prediction_original': pred,
        'actual': actual,
        'scenarios': [
            {
                'scenario': 'no_snap',
                'was_active': snap_was_active,
                'prediction_cf': pred + snap_delta,
                'delta': snap_delta,
                'delta_pct': snap_delta / pred * 100 if pred else 0,
            },
            {
                'scenario': 'no_event',
                'was_active': False,
                'prediction_cf': pred,
                'delta': 0,
                'delta_pct': 0,
            },
            {
                'scenario': 'no_price_change',
                'was_active': False,
                'prediction_cf': pred,
                'delta': 0,
                'delta_pct': 0,
            },
        ],
    })


def _make_contrastive_payload(bad_week='2013-01-05', good_week='2012-01-07'):
    return json.dumps({
        'bad_week': bad_week,
        'good_week': good_week,
        'good_week_mape': 5.0,
        'seasonality_matched': True,
        'top_diffs': [
            {'feature': 'lag_1', 'shap_diff': 0.9, 'bad_value': 50.0, 'good_value': 6.0,
             'bad_shap': 0.8, 'good_shap': 0.05},
        ],
    })


@pytest.fixture
def seeded_db(tmp_path):
    """Fresh DB with realistic bad-week data across several items/weeks."""
    db_path = str(tmp_path / 'test_insights.db')
    conn = get_conn(db_path)

    # Seed evaluations
    conn.executemany(
        'INSERT OR REPLACE INTO evaluations (week_id, item_id, h1_mape, h1_mae, is_bad_week, mape_zscore) VALUES (?,?,?,?,?,?)',
        [
            ('2013-01-05', 'ITEM_001', 120.0, 40.0, 1, 2.1),
            ('2013-01-05', 'ITEM_002', 95.0,  30.0, 1, 2.1),
            ('2013-01-05', 'ITEM_003', 80.0,  25.0, 1, 2.1),
            ('2013-02-02', 'ITEM_001', 10.0,   3.0, 0, 0.2),
            ('2013-02-02', 'ITEM_002',  8.0,   2.0, 0, 0.2),
            ('2013-02-02', 'ITEM_003',  7.0,   1.5, 0, 0.2),
        ],
    )

    # Seed forecasts
    conn.executemany(
        'INSERT OR REPLACE INTO forecasts (week_id, item_id, h1, trained_at) VALUES (?,?,?,?)',
        [
            ('2013-01-05', 'ITEM_001', 48.0, '2013-01-01'),
            ('2013-01-05', 'ITEM_002', 38.0, '2013-01-01'),
            ('2013-01-05', 'ITEM_003', 28.0, '2013-01-01'),
        ],
    )

    # Seed xai_results — 3 items, all over-forecasts, lag_1 dominates
    for item, lag1 in [('ITEM_001', 50.0), ('ITEM_002', 40.0), ('ITEM_003', 30.0)]:
        conn.execute(
            'INSERT OR REPLACE INTO xai_results (week_id, item_id, xai_type, payload) VALUES (?,?,?,?)',
            ('2013-01-05', item, 'shap', _make_shap_payload(
                direction='over', lag1_val=lag1, actual=5.0, pred=lag1 - 2,
            )),
        )
        conn.execute(
            'INSERT OR REPLACE INTO xai_results (week_id, item_id, xai_type, payload) VALUES (?,?,?,?)',
            ('2013-01-05', item, 'counterfactual', _make_cf_payload(
                pred=lag1 - 2, snap_was_active=True, snap_delta=-8.0,
            )),
        )
        conn.execute(
            'INSERT OR REPLACE INTO xai_results (week_id, item_id, xai_type, payload) VALUES (?,?,?,?)',
            ('2013-01-05', item, 'contrastive', _make_contrastive_payload()),
        )

    # Seed external signals
    conn.execute(
        'INSERT OR REPLACE INTO external_signals (week, temp_mean, temp_max, temp_min, precip, heat_days, gas_price, consumer_sentiment) '
        'VALUES (?,?,?,?,?,?,?,?)',
        ('2013-01-05', 38.0, 41.0, 32.0, 2.0, 5, 4.60, 62.0),  # heat_wave + gas_spike + crisis_low
    )

    conn.commit()
    return conn


# ── Detector tests ────────────────────────────────────────────────────────────

class TestOverForecastBias:
    def test_fires_when_all_over(self, seeded_db):
        result = detect_over_forecast_bias(seeded_db)
        assert result is not None
        assert result.finding_type == 'over_forecast_bias'
        assert result.evidence['pct_over'] == 100.0
        assert result.score == 1.0

    def test_returns_none_when_no_shap(self, db_conn):
        result = detect_over_forecast_bias(db_conn)
        assert result is None

    def test_evidence_has_required_keys(self, seeded_db):
        result = detect_over_forecast_bias(seeded_db)
        assert result is not None
        for key in ('total_payloads', 'over_count', 'under_count', 'pct_over', 'implication'):
            assert key in result.evidence


class TestDominantDriver:
    def test_fires_when_lag1_dominates(self, seeded_db):
        result = detect_dominant_driver(seeded_db)
        assert result is not None
        assert result.finding_type == 'dominant_driver'
        dominant_features = [f['feature'] for f in result.evidence['dominant_features']]
        assert 'lag_1' in dominant_features or 'rolling_4_mean' in dominant_features

    def test_evidence_has_all_feature_counts(self, seeded_db):
        result = detect_dominant_driver(seeded_db)
        assert result is not None
        assert 'all_feature_counts' in result.evidence
        assert len(result.evidence['all_feature_counts']) > 0


class TestDemandCliff:
    def test_fires_when_lag1_far_exceeds_actual(self, seeded_db):
        result = detect_demand_cliff(seeded_db)
        assert result is not None
        assert result.finding_type == 'demand_cliff'
        assert result.evidence['n_cliff_items'] == 3

    def test_examples_have_trajectory_info(self, seeded_db):
        result = detect_demand_cliff(seeded_db)
        assert result is not None
        for ex in result.evidence['top_examples']:
            assert 'lag_1_value' in ex
            assert 'actual' in ex
            assert 'cliff_ratio' in ex

    def test_returns_none_when_no_shap(self, db_conn):
        result = detect_demand_cliff(db_conn)
        assert result is None


class TestExternalCoincidence:
    def test_fires_on_notable_conditions(self, seeded_db):
        result = detect_external_coincidence(seeded_db)
        assert result is not None
        assert result.finding_type == 'external_coincidence'
        assert len(result.evidence['notable_weeks']) > 0

    def test_caveat_present(self, seeded_db):
        result = detect_external_coincidence(seeded_db)
        assert result is not None
        assert 'caveat' in result.evidence
        assert 'CORRELATIONS' in result.evidence['caveat']

    def test_returns_none_when_no_external(self, db_conn):
        result = detect_external_coincidence(db_conn)
        assert result is None


class TestCounterfactualMaterial:
    def test_fires_when_snap_material(self, seeded_db):
        result = detect_counterfactual_material(seeded_db)
        assert result is not None
        assert result.finding_type == 'counterfactual_material'

    def test_scenario_impacts_have_required_keys(self, seeded_db):
        result = detect_counterfactual_material(seeded_db)
        assert result is not None
        for s in result.evidence['scenario_impacts']:
            assert 'scenario' in s
            assert 'pct_items' in s
            assert 'avg_delta_pct' in s


class TestContrastiveGap:
    def test_fires_when_contrastive_exists(self, seeded_db):
        result = detect_contrastive_gap(seeded_db)
        assert result is not None
        assert result.finding_type == 'contrastive_gap'

    def test_coverage_pct_correct(self, seeded_db):
        result = detect_contrastive_gap(seeded_db)
        assert result is not None
        # 3 contrastive / 3 shap = 100%
        assert result.evidence['coverage_pct'] == 100.0


class TestRunAllDetectors:
    def test_returns_list(self, seeded_db):
        results = run_all_detectors(seeded_db)
        assert isinstance(results, list)

    def test_sorted_by_score_desc(self, seeded_db):
        results = run_all_detectors(seeded_db)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_finding_types_present(self, seeded_db):
        results = run_all_detectors(seeded_db)
        types = {r.finding_type for r in results}
        assert 'over_forecast_bias' in types
        assert 'demand_cliff' in types


# ── Tool tests ────────────────────────────────────────────────────────────────

class TestReadTools:
    def test_read_forecast_accuracy(self, seeded_db):
        acc = read_forecast_accuracy(seeded_db)
        assert acc['total_weeks'] == 2
        assert acc['n_bad_weeks'] == 1
        assert acc['n_good_weeks'] == 1

    def test_read_bad_weeks(self, seeded_db):
        bw = read_bad_weeks(seeded_db)
        assert len(bw) == 1
        assert bw[0]['week_id'] == '2013-01-05'

    def test_read_xai_findings_filters_by_type(self, seeded_db):
        shap_rows = read_xai_findings(seeded_db, xai_type='shap')
        assert all(r['xai_type'] == 'shap' for r in shap_rows)
        assert len(shap_rows) == 3

    def test_read_external_signals_returns_labels(self, seeded_db):
        sig = read_external_signals(seeded_db, '2013-01-05')
        assert sig is not None
        assert sig['weather_label'] == 'heat_wave'
        assert sig['gas_label'] == 'historic_spike'
        assert sig['sentiment_label'] == 'crisis_low'

    def test_read_external_signals_missing_week(self, seeded_db):
        sig = read_external_signals(seeded_db, '2099-01-01')
        assert sig is None

    def test_read_recurring_drivers(self, seeded_db):
        drivers = read_recurring_drivers(seeded_db)
        assert len(drivers) > 0
        assert drivers[0]['count'] >= drivers[-1]['count']
        # lag_1 and rolling_4_mean should appear
        features = {d['feature'] for d in drivers}
        assert 'lag_1' in features


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestSchemas:
    def test_candidate_finding_fields(self):
        c = CandidateFinding(
            finding_id='test', finding_type='over_forecast_bias',
            score=0.9, summary='test summary',
        )
        assert c.finding_id == 'test'
        assert c.evidence == {}

    def test_ledger_row_fields(self):
        row = LedgerRow(
            finding_id='test', finding_type='demand_cliff',
            status='accepted', confidence='high',
            evidence={'n': 5}, hypothesis={'headline': 'h'},
            critic_notes='ok',
        )
        assert row.status == 'accepted'
        assert row.hypothesis['headline'] == 'h'


# ── LLM no-key fallback ───────────────────────────────────────────────────────

class TestLLMClient:
    def test_raises_on_missing_key(self, monkeypatch):
        monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
        from xai_forecast.insights.llm_client import DeepSeekClient
        with pytest.raises(RuntimeError, match='DEEPSEEK_API_KEY'):
            DeepSeekClient()

    def test_async_methods_exist(self, monkeypatch):
        """Both sync and async interfaces must be present on the client."""
        import inspect
        monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
        from xai_forecast.insights.llm_client import DeepSeekClient
        client = DeepSeekClient()
        assert callable(client.call_flash)
        assert callable(client.call_pro)
        assert callable(client.acall_flash)
        assert callable(client.acall_pro)
        assert inspect.iscoroutinefunction(client.acall_flash)
        assert inspect.iscoroutinefunction(client.acall_pro)
