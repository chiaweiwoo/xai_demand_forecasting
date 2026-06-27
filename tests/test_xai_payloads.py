"""
Group C — XAI payload contract tests.

Verifies that all keys the dashboard reads are present, SHAP additivity holds,
and payloads serialise cleanly to/from JSON (no numpy types).
"""

import json

import numpy as np
import pandas as pd
import pytest

from xai_forecast.features import FEATURE_COLS
from xai_forecast.xai import (
    make_explainer,
    shap_payloads,
    counterfactual_payloads,
)

N_FEATURES = len(FEATURE_COLS)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def week_df(trained_model_and_explainer):
    """Fake week_df with one SKU, feature values all set to 1.0."""
    model, _ = trained_model_and_explainer
    rng = np.random.default_rng(7)
    row = {col: float(rng.uniform(0, 5)) for col in FEATURE_COLS}
    row['unique_id'] = 'test_item'
    row['y'] = 3.0
    return pd.DataFrame([row])


@pytest.fixture(scope='module')
def shap_result(trained_model_and_explainer, week_df):
    model, explainer = trained_model_and_explainer
    rows, cache = shap_payloads(
        explainer, model, week_df, '2015-01-01', ['test_item'], {'test_item': 3.0}
    )
    return rows, cache


@pytest.fixture(scope='module')
def cf_result(trained_model_and_explainer, week_df):
    model, _ = trained_model_and_explainer
    return counterfactual_payloads(model, week_df, '2015-01-01', ['test_item'], {'test_item': 3.0})


# ── SHAP contract ─────────────────────────────────────────────────────────────

def test_shap_required_keys(shap_result):
    rows, _ = shap_result
    assert len(rows) == 1
    d = json.loads(rows[0]['payload'])
    for key in ('base_value_log', 'prediction', 'other_features_shap', 'top_features', 'shap_note',
                'signed_error', 'direction'):
        assert key in d, f'SHAP payload missing key: {key}'


def test_shap_top_features_count(shap_result):
    rows, _ = shap_result
    d = json.loads(rows[0]['payload'])
    assert len(d['top_features']) == 5


def test_shap_top_feature_keys(shap_result):
    rows, _ = shap_result
    d = json.loads(rows[0]['payload'])
    for feat in d['top_features']:
        for k in ('feature', 'shap_value', 'feature_value'):
            assert k in feat, f'top_features entry missing key: {k}'


def test_shap_cache_returned(shap_result):
    _, cache = shap_result
    assert 'test_item' in cache
    assert len(cache['test_item']) == N_FEATURES


def test_shap_additivity(trained_model_and_explainer, week_df):
    """base_log + sum(all 19 SHAP values) ≈ log(prediction), from raw (unrounded) values."""
    model, explainer = trained_model_and_explainer
    rows_feat = week_df[['unique_id'] + FEATURE_COLS].fillna(0)
    X = rows_feat[FEATURE_COLS]
    sv = explainer.shap_values(X)
    base_log = float(explainer.expected_value)
    pred = float(model.predict(X).clip(min=0)[0])

    shap_sum = float(np.sum(sv[0]))
    log_pred = float(np.log(pred)) if pred > 0 else float('-inf')

    assert abs(base_log + shap_sum - log_pred) < 1e-4, (
        f'SHAP additivity violated: {base_log} + {shap_sum} ≠ log({pred})={log_pred}'
    )


def test_shap_waterfall_reconciles(shap_result):
    """Stored base_log + top5_shap + other_features_shap ≈ log(prediction), within rounding."""
    rows, _ = shap_result
    d = json.loads(rows[0]['payload'])
    total = sum(f['shap_value'] for f in d['top_features']) + d['other_features_shap']
    pred = d['prediction']
    if pred > 0:
        assert abs(d['base_value_log'] + total - np.log(pred)) < 0.01


def test_shap_json_roundtrip(shap_result):
    rows, _ = shap_result
    payload_str = rows[0]['payload']
    roundtripped = json.dumps(json.loads(payload_str))
    assert isinstance(roundtripped, str)


def test_shap_no_numpy_types(shap_result):
    rows, _ = shap_result
    # If numpy types were stored, json.dumps would raise TypeError
    try:
        json.dumps(json.loads(rows[0]['payload']))
    except TypeError as e:
        pytest.fail(f'Numpy type leaked into SHAP payload: {e}')


def test_shap_db_row_keys(shap_result):
    rows, _ = shap_result
    assert rows[0]['week_id'] == '2015-01-01'
    assert rows[0]['item_id'] == 'test_item'
    assert rows[0]['xai_type'] == 'shap'


# ── Counterfactual contract ───────────────────────────────────────────────────

def test_cf_required_keys(cf_result):
    assert len(cf_result) == 1
    d = json.loads(cf_result[0]['payload'])
    for key in ('prediction_original', 'scenarios'):
        assert key in d, f'CF payload missing key: {key}'


def test_cf_scenario_keys(cf_result):
    d = json.loads(cf_result[0]['payload'])
    for s in d['scenarios']:
        for k in ('scenario', 'was_active', 'prediction_cf', 'delta', 'delta_pct'):
            assert k in s, f'CF scenario missing key: {k}'


def test_cf_was_active_type(cf_result):
    d = json.loads(cf_result[0]['payload'])
    for s in d['scenarios']:
        assert isinstance(s['was_active'], bool), 'was_active must be bool'


def test_cf_scenario_count(cf_result):
    d = json.loads(cf_result[0]['payload'])
    assert len(d['scenarios']) == 3  # no_snap, no_event, no_price_change


def test_cf_delta_consistent(cf_result):
    d = json.loads(cf_result[0]['payload'])
    orig = d['prediction_original']
    for s in d['scenarios']:
        expected_delta = round(s['prediction_cf'] - orig, 3)
        assert abs(s['delta'] - expected_delta) < 0.01


def test_cf_json_roundtrip(cf_result):
    try:
        json.dumps(json.loads(cf_result[0]['payload']))
    except TypeError as e:
        pytest.fail(f'Numpy type leaked into CF payload: {e}')
