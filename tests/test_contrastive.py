"""
Group C-ext — contrastive XAI tests.
Previously had zero coverage; this file addresses the gap.
"""

import json

import numpy as np
import pandas as pd
import pytest

from xai_forecast.features import FEATURE_COLS
from xai_forecast.xai import contrastive_payloads, shap_payloads


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_all_evals(weeks_mapes: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build a minimal all_evals_df: [(forecast_week, unique_id, mape)]."""
    return pd.DataFrame(
        [{'forecast_week': w, 'unique_id': uid, 'mape': m} for w, uid, m in weeks_mapes]
    )


def _make_week_df(uid: str, woy_week: str, seed: int = 0) -> pd.DataFrame:
    """Single-row week_df for one SKU."""
    rng = np.random.default_rng(seed)
    row = {col: float(rng.uniform(0, 5)) for col in FEATURE_COLS}
    row['unique_id'] = uid
    row['y'] = 3.0
    # Set week_of_year from the date
    row['week_of_year'] = pd.Timestamp(woy_week).isocalendar()[1]
    return pd.DataFrame([row])


# ── Same-WOY selection ────────────────────────────────────────────────────────

def test_contrastive_picks_same_woy(trained_model_and_explainer, db_conn):
    """contrastive should match a good week with the same ISO week-of-year."""
    model, explainer = trained_model_and_explainer
    bad_week = '2015-01-03'  # ISO WOY 1
    ref_week  = '2014-01-04'  # same WOY 1
    other_ref = '2014-04-05'  # WOY 14 — should NOT be picked

    uid = 'CA_1_001_TX_1'
    all_evals = _make_all_evals([
        (ref_week, uid, 5.0),    # WOY 1, good
        (other_ref, uid, 3.0),   # WOY 14, good but different season
    ])

    # Seed the DB with feature rows for both ref weeks
    for wk, seed in [(ref_week, 1), (other_ref, 2)]:
        week_row = _make_week_df(uid, wk, seed=seed)
        df = week_row[['unique_id'] + FEATURE_COLS].copy()
        df['week'] = wk
        df['y'] = 2.0
        df.to_sql('features', db_conn, if_exists='append', index=False)
    db_conn.commit()

    bad_week_df = _make_week_df(uid, bad_week, seed=99)
    results = contrastive_payloads(
        explainer, bad_week_df, bad_week, [uid], all_evals, db_conn
    )
    assert len(results) == 1
    payload = json.loads(results[0]['payload'])
    assert payload['good_week'] == ref_week, (
        f"Expected same-WOY ref {ref_week}, got {payload['good_week']}"
    )
    assert payload['seasonality_matched'] is True


def test_contrastive_skips_when_no_same_woy(trained_model_and_explainer, db_conn):
    """When no same-WOY good week exists, contrastive should return empty (no fallback)."""
    model, explainer = trained_model_and_explainer
    bad_week = '2015-01-03'   # WOY 1
    ref_week  = '2014-04-05'  # WOY 14 — different season, should not be used

    uid = 'CA_1_001_TX_1'
    all_evals = _make_all_evals([(ref_week, uid, 5.0)])

    week_row = _make_week_df(uid, ref_week, seed=1)
    df = week_row[['unique_id'] + FEATURE_COLS].copy()
    df['week'] = ref_week
    df['y'] = 2.0
    df.to_sql('features', db_conn, if_exists='append', index=False)
    db_conn.commit()

    bad_week_df = _make_week_df(uid, bad_week, seed=99)
    results = contrastive_payloads(
        explainer, bad_week_df, bad_week, [uid], all_evals, db_conn
    )
    assert results == [], 'Should return empty when no same-WOY good week exists'


def test_contrastive_skips_when_no_good_weeks(trained_model_and_explainer, db_conn):
    """Item with all MAPE >= 15 → always skip."""
    model, explainer = trained_model_and_explainer
    bad_week = '2015-01-03'
    uid = 'CA_1_001_TX_1'
    all_evals = _make_all_evals([
        ('2014-01-04', uid, 80.0),  # bad mape
        ('2014-01-11', uid, 50.0),
    ])
    bad_week_df = _make_week_df(uid, bad_week)
    results = contrastive_payloads(
        explainer, bad_week_df, bad_week, [uid], all_evals, db_conn
    )
    assert results == []


# ── Payload contract ──────────────────────────────────────────────────────────

def test_contrastive_payload_keys(trained_model_and_explainer, db_conn):
    """Payload must have all keys the dashboard reads."""
    model, explainer = trained_model_and_explainer
    bad_week = '2015-01-03'
    ref_week  = '2014-01-04'
    uid = 'CA_1_001_TX_1'
    all_evals = _make_all_evals([(ref_week, uid, 5.0)])

    week_row = _make_week_df(uid, ref_week, seed=1)
    df = week_row[['unique_id'] + FEATURE_COLS].copy()
    df['week'] = ref_week
    df['y'] = 2.0
    df.to_sql('features', db_conn, if_exists='append', index=False)
    db_conn.commit()

    bad_week_df = _make_week_df(uid, bad_week, seed=99)
    results = contrastive_payloads(
        explainer, bad_week_df, bad_week, [uid], all_evals, db_conn
    )
    assert results, 'Expected one contrastive result'
    payload = json.loads(results[0]['payload'])
    for key in ('bad_week', 'good_week', 'good_week_mape', 'seasonality_matched', 'top_diffs'):
        assert key in payload, f'Missing key: {key}'

    for diff in payload['top_diffs']:
        for k in ('feature', 'shap_diff', 'bad_value', 'good_value', 'bad_shap', 'good_shap'):
            assert k in diff, f'top_diffs entry missing key: {k}'


def test_contrastive_shap_diff_math(trained_model_and_explainer, db_conn):
    """shap_diff must equal bad_shap - good_shap for every entry."""
    model, explainer = trained_model_and_explainer
    bad_week = '2015-01-03'
    ref_week  = '2014-01-04'
    uid = 'CA_1_001_TX_1'
    all_evals = _make_all_evals([(ref_week, uid, 5.0)])

    week_row = _make_week_df(uid, ref_week, seed=1)
    df = week_row[['unique_id'] + FEATURE_COLS].copy()
    df['week'] = ref_week
    df['y'] = 2.0
    df.to_sql('features', db_conn, if_exists='append', index=False)
    db_conn.commit()

    bad_week_df = _make_week_df(uid, bad_week, seed=99)
    results = contrastive_payloads(
        explainer, bad_week_df, bad_week, [uid], all_evals, db_conn
    )
    assert results
    payload = json.loads(results[0]['payload'])
    for d in payload['top_diffs']:
        expected = round(d['bad_shap'] - d['good_shap'], 4)
        assert abs(d['shap_diff'] - expected) < 0.001, (
            f"shap_diff {d['shap_diff']} != bad_shap - good_shap = {expected}"
        )


def test_contrastive_cache_vs_fresh_shap(trained_model_and_explainer, db_conn):
    """SHAP values from bad_shap_cache must match values computed fresh."""
    model, explainer = trained_model_and_explainer
    bad_week = '2015-01-03'
    ref_week  = '2014-01-04'
    uid = 'CA_1_001_TX_1'
    all_evals = _make_all_evals([(ref_week, uid, 5.0)])

    week_row = _make_week_df(uid, ref_week, seed=1)
    df = week_row[['unique_id'] + FEATURE_COLS].copy()
    df['week'] = ref_week
    df['y'] = 2.0
    df.to_sql('features', db_conn, if_exists='append', index=False)
    db_conn.commit()

    bad_week_df = _make_week_df(uid, bad_week, seed=99)

    # With cache (from shap_payloads)
    _, shap_cache = shap_payloads(explainer, model, bad_week_df, bad_week, [uid], {uid: 3.0})
    results_cached = contrastive_payloads(
        explainer, bad_week_df, bad_week, [uid], all_evals, db_conn, bad_shap_cache=shap_cache
    )

    # Without cache (contrastive recomputes from scratch) — use a fresh db connection
    import os, tempfile
    from xai_forecast.db import get_conn
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    db_conn2 = get_conn(path)
    df.to_sql('features', db_conn2, if_exists='append', index=False)
    db_conn2.commit()

    results_fresh = contrastive_payloads(
        explainer, bad_week_df, bad_week, [uid], all_evals, db_conn2
    )
    db_conn2.close()
    os.unlink(path)

    assert results_cached and results_fresh
    cached_diffs = json.loads(results_cached[0]['payload'])['top_diffs']
    fresh_diffs  = json.loads(results_fresh[0]['payload'])['top_diffs']

    # Same features in same order, same values
    for c, f in zip(cached_diffs, fresh_diffs):
        assert c['feature'] == f['feature']
        assert abs(c['bad_shap'] - f['bad_shap']) < 1e-4, (
            f"bad_shap mismatch for {c['feature']}: cache={c['bad_shap']} fresh={f['bad_shap']}"
        )


def test_contrastive_reference_is_most_recent(trained_model_and_explainer, db_conn):
    """When multiple same-WOY good weeks exist, picks the most recent one."""
    model, explainer = trained_model_and_explainer
    bad_week = '2016-01-02'  # WOY 53 or 1 — let's use a week with clear WOY
    # Use a bad week in 2016 and two ref weeks in same WOY but different years
    bad_week = '2015-01-03'  # WOY 1
    ref_old  = '2013-01-05'  # WOY 1 (older)
    ref_new  = '2014-01-04'  # WOY 1 (newer — should be picked)
    uid = 'CA_1_001_TX_1'
    all_evals = _make_all_evals([
        (ref_old, uid, 4.0),
        (ref_new, uid, 6.0),
    ])

    for wk, seed in [(ref_old, 1), (ref_new, 2)]:
        row = _make_week_df(uid, wk, seed=seed)
        df = row[['unique_id'] + FEATURE_COLS].copy()
        df['week'] = wk
        df['y'] = 2.0
        df.to_sql('features', db_conn, if_exists='append', index=False)
    db_conn.commit()

    bad_week_df = _make_week_df(uid, bad_week, seed=99)
    results = contrastive_payloads(
        explainer, bad_week_df, bad_week, [uid], all_evals, db_conn
    )
    assert results
    payload = json.loads(results[0]['payload'])
    assert payload['good_week'] == ref_new, (
        f"Expected most recent {ref_new}, got {payload['good_week']}"
    )
