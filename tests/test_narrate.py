"""
Tests for xai_forecast/narrate.py.

All tests run without a real API key or network access.
The DeepSeekNarrator._client is injected directly to mock the OpenAI call.
"""

import json
import os
from unittest.mock import MagicMock

import pytest

from xai_forecast.narrate import (
    DeepSeekNarrator,
    WEEK_NARRATIVE_PROMPT,
    ITEM_NARRATIVE_PROMPT,
    EXECUTIVE_NARRATIVE_PROMPT,
    _grounding_check,
    build_week_dossier,
    build_item_dossier,
    build_executive_dossier,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _shap_row(week_id='2015-01-03', item_id='CA_1_001'):
    return {
        'week_id': week_id,
        'item_id': item_id,
        'xai_type': 'shap',
        'payload': json.dumps({
            'base_value_log': 0.5,
            'prediction': 4.2,
            'actual': 3.0,
            'error_pct': 40.0,
            'signed_error': 40.0,
            'direction': 'over',
            'other_features_shap': -0.1,
            'top_features': [
                {'feature': 'lag_1', 'shap_value': 0.8, 'feature_value': 3.0},
                {'feature': 'snap', 'shap_value': 0.3, 'feature_value': 1.0},
                {'feature': 'rolling_4_mean', 'shap_value': 0.2, 'feature_value': 2.5},
                {'feature': 'sell_price', 'shap_value': -0.15, 'feature_value': 9.99},
                {'feature': 'month', 'shap_value': 0.05, 'feature_value': 1.0},
            ],
        }),
    }


def _cf_row(week_id='2015-01-03', item_id='CA_1_001'):
    return {
        'week_id': week_id,
        'item_id': item_id,
        'xai_type': 'counterfactual',
        'payload': json.dumps({
            'prediction_original': 4.2,
            'actual': 3.0,
            'scenarios': [
                {'scenario': 'no_snap', 'was_active': True, 'prediction_cf': 3.5, 'delta': -0.7, 'delta_pct': -16.7},
                {'scenario': 'no_event', 'was_active': False, 'prediction_cf': 4.2, 'delta': 0.0, 'delta_pct': 0.0},
                {'scenario': 'no_price_change', 'was_active': True, 'prediction_cf': 4.0, 'delta': -0.2, 'delta_pct': -4.8},
            ],
        }),
    }


def _cont_row(week_id='2015-01-03', item_id='CA_1_001'):
    return {
        'week_id': week_id,
        'item_id': item_id,
        'xai_type': 'contrastive',
        'payload': json.dumps({
            'bad_week': week_id,
            'good_week': '2014-01-04',
            'good_week_mape': 8.5,
            'seasonality_matched': True,
            'top_diffs': [
                {'feature': 'lag_1', 'shap_diff': 0.4, 'bad_value': 3.0, 'good_value': 1.5,
                 'bad_shap': 0.8, 'good_shap': 0.4},
            ],
        }),
    }


def _make_narrator_with_mock(response: dict) -> DeepSeekNarrator:
    """Build a narrator with a mock client returning the given response dict."""
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(response)
    mock_client.chat.completions.create.return_value.choices = [MagicMock(message=mock_msg)]

    narrator = DeepSeekNarrator.__new__(DeepSeekNarrator)
    narrator._key = 'test-key'
    narrator.model_id = 'deepseek-v4-flash'
    narrator._base_url = 'https://api.deepseek.com'
    narrator._client = mock_client
    return narrator


# ── build_week_dossier ────────────────────────────────────────────────────────

def test_build_week_dossier_top_features_sorted():
    rows = [_shap_row(), _shap_row('2015-01-03', 'CA_1_002')]
    dossier = build_week_dossier('2015-01-03', rows, wmape_zscore=2.5, n_items_in_week=50)
    feats = dossier['top_features']
    assert feats[0]['feature'] == 'lag_1', 'Highest mean |SHAP| feature first'
    assert feats[0]['mean_abs_shap'] >= feats[1]['mean_abs_shap']


def test_build_week_dossier_features_list():
    rows = [_shap_row()]
    dossier = build_week_dossier('2015-01-03', rows, wmape_zscore=1.8, n_items_in_week=30)
    assert 'features' in dossier
    assert 'lag_1' in dossier['features']


def test_build_week_dossier_empty_rows():
    dossier = build_week_dossier('2015-01-03', [], wmape_zscore=None, n_items_in_week=0)
    assert dossier['top_features'] == []
    assert dossier['features'] == []
    assert dossier['wmape_zscore'] is None


def test_build_week_dossier_pct_of_top_features_sums_to_100():
    rows = [_shap_row()]
    dossier = build_week_dossier('2015-01-03', rows, wmape_zscore=2.0, n_items_in_week=20)
    total_pct = sum(f['pct_of_top_features'] for f in dossier['top_features'])
    assert abs(total_pct - 100.0) < 1.0, f'pct_of_top_features should sum to ~100, got {total_pct}'


# ── build_item_dossier ────────────────────────────────────────────────────────

def test_build_item_dossier_full_payload():
    shap_p = json.loads(_shap_row()['payload'])
    cf_p   = json.loads(_cf_row()['payload'])
    cont_p = json.loads(_cont_row()['payload'])
    dossier = build_item_dossier('2015-01-03', 'CA_1_001', shap_p, cf_p, cont_p)
    assert dossier['direction'] == 'over'
    assert 'lag_1' in dossier['features']
    assert 'contrastive' in dossier
    assert dossier['contrastive']['good_week'] == '2014-01-04'


def test_build_item_dossier_cf_filters_inactive():
    shap_p = json.loads(_shap_row()['payload'])
    cf_p   = json.loads(_cf_row()['payload'])
    dossier = build_item_dossier('2015-01-03', 'CA_1_001', shap_p, cf_p, None)
    # no_event is was_active=False → must not appear in active_counterfactuals
    active_scenarios = [c['scenario'] for c in dossier.get('active_counterfactuals', [])]
    assert 'no_event' not in active_scenarios


def test_build_item_dossier_no_shap():
    dossier = build_item_dossier('2015-01-03', 'CA_1_001', None, None, None)
    assert dossier['features'] == []
    assert 'prediction' not in dossier


# ── build_executive_dossier ───────────────────────────────────────────────────

def test_build_executive_dossier_structure():
    drivers = [
        {'feature': 'lag_1', 'count': 80, 'pct_payloads': 70.0},
        {'feature': 'snap', 'count': 50, 'pct_payloads': 44.0},
    ]
    dossier = build_executive_dossier(drivers, n_bad_weeks=15, n_total_weeks=120)
    assert dossier['n_bad_weeks'] == 15
    assert dossier['bad_week_rate_pct'] == pytest.approx(12.5, abs=0.1)
    assert dossier['features'] == ['lag_1', 'snap']


def test_build_executive_dossier_zero_total():
    dossier = build_executive_dossier([], n_bad_weeks=0, n_total_weeks=0)
    assert dossier['bad_week_rate_pct'] == 0


# ── _grounding_check ──────────────────────────────────────────────────────────

def test_grounding_check_passes_valid_feature():
    narr = {'primary_driver': 'lag_1', 'headline': '', 'body': '', 'confidence': 'high'}
    dossier = {'features': ['lag_1', 'snap', 'rolling_4_mean']}
    assert _grounding_check(narr, dossier) is True


def test_grounding_check_fails_hallucinated_feature():
    narr = {'primary_driver': 'weather_temperature', 'headline': '', 'body': '', 'confidence': 'high'}
    dossier = {'features': ['lag_1', 'snap', 'rolling_4_mean']}
    assert _grounding_check(narr, dossier) is False


def test_grounding_check_passes_when_no_features_in_dossier():
    narr = {'primary_driver': 'anything'}
    dossier = {}  # can't verify — assume OK
    assert _grounding_check(narr, dossier) is True


def test_grounding_check_fails_empty_primary_driver():
    narr = {'primary_driver': ''}
    dossier = {'features': ['lag_1']}
    assert _grounding_check(narr, dossier) is False


# ── DeepSeekNarrator.available ────────────────────────────────────────────────

def test_narrator_unavailable_without_key():
    orig = os.environ.pop('DEEPSEEK_API_KEY', None)
    try:
        narrator = DeepSeekNarrator()
        assert not narrator.available
    finally:
        if orig is not None:
            os.environ['DEEPSEEK_API_KEY'] = orig


def test_narrator_generate_returns_none_without_key():
    orig = os.environ.pop('DEEPSEEK_API_KEY', None)
    try:
        narrator = DeepSeekNarrator()
        result = narrator.generate(WEEK_NARRATIVE_PROMPT, {'features': ['lag_1']})
        assert result is None
    finally:
        if orig is not None:
            os.environ['DEEPSEEK_API_KEY'] = orig


# ── DeepSeekNarrator.generate (mocked) ───────────────────────────────────────

def test_generate_returns_schema_compliant_dict():
    response = {
        'headline': 'Lag features drove a 40% over-forecast',
        'body': 'Recent sales trend shifted sharply but the model relied on stale lag values.',
        'primary_driver': 'lag_1',
        'confidence': 'high',
    }
    rows = [_shap_row()]
    dossier = build_week_dossier('2015-01-03', rows, wmape_zscore=2.5, n_items_in_week=50)
    narrator = _make_narrator_with_mock(response)
    result = narrator.generate(WEEK_NARRATIVE_PROMPT, dossier)

    assert result is not None
    for key in ('headline', 'body', 'primary_driver', 'confidence', 'model'):
        assert key in result, f'Missing key: {key}'
    assert result['primary_driver'] == 'lag_1'
    assert result['model'] == 'deepseek-v4-flash'


def test_generate_flags_grounding_violation():
    response = {
        'headline': 'Weather caused the error',
        'body': 'External temperature data misled the model.',
        'primary_driver': 'weather_temperature',  # hallucinated — not in dossier features
        'confidence': 'high',
    }
    rows = [_shap_row()]
    dossier = build_week_dossier('2015-01-03', rows, wmape_zscore=2.5, n_items_in_week=50)
    narrator = _make_narrator_with_mock(response)
    result = narrator.generate(WEEK_NARRATIVE_PROMPT, dossier)

    assert result is not None
    assert result['confidence'] == 'low', 'Grounding failure must downgrade confidence to low'
    assert result.get('grounding_warning') is True


def test_generate_returns_none_on_missing_schema_key():
    # LLM omits required 'confidence' key
    response = {'headline': 'X', 'body': 'Y', 'primary_driver': 'lag_1'}
    rows = [_shap_row()]
    dossier = build_week_dossier('2015-01-03', rows, wmape_zscore=2.5, n_items_in_week=50)
    narrator = _make_narrator_with_mock(response)
    result = narrator.generate(WEEK_NARRATIVE_PROMPT, dossier)
    assert result is None


def test_generate_returns_none_on_exception():
    narrator = DeepSeekNarrator.__new__(DeepSeekNarrator)
    narrator._key = 'test-key'
    narrator.model_id = 'deepseek-v4-flash'
    narrator._base_url = 'https://api.deepseek.com'
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception('network error')
    narrator._client = mock_client

    result = narrator.generate(WEEK_NARRATIVE_PROMPT, {'features': ['lag_1']})
    assert result is None


def test_generate_payload_json_round_trip():
    """Narrative stored as JSON and parsed back preserves all keys."""
    response = {
        'headline': 'Promotion week inflated the forecast by 17%',
        'body': 'SNAP promotion on this week caused a spike in demand the model over-estimated.',
        'primary_driver': 'snap',
        'confidence': 'medium',
    }
    rows = [_shap_row()]
    dossier = build_week_dossier('2015-01-03', rows, wmape_zscore=2.0, n_items_in_week=10)
    narrator = _make_narrator_with_mock(response)
    result = narrator.generate(WEEK_NARRATIVE_PROMPT, dossier)

    assert result is not None
    stored = json.dumps(result)
    recovered = json.loads(stored)
    assert recovered['primary_driver'] == 'snap'
    assert recovered['confidence'] == 'medium'
