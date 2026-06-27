"""
Group A — feature engineering tests.
All tests use synthetic data; no SQLite required.
"""

import numpy as np
import pandas as pd
import pytest

from xai_forecast.features import compute_features, FEATURE_COLS, LAG_WEEKS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _single_sku(n_weeks: int = 20, y_start: int = 1) -> pd.DataFrame:
    """One SKU with y = y_start, y_start+1, … for deterministic tests."""
    weeks = pd.date_range('2014-01-04', periods=n_weeks, freq='7D').strftime('%Y-%m-%d').tolist()
    return pd.DataFrame({
        'week': weeks,
        'unique_id': 'A',
        'y': [float(y_start + i) for i in range(n_weeks)],
        'snap': 0, 'has_event': 0, 'event_type_enc': 0,
        'sell_price': 2.0,
        'dept_mean_sales': 5.0, 'cat_mean_sales': 10.0,
    })


# ── lag_n correctness ────────────────────────────────────────────────────────

@pytest.mark.parametrize('lag', [1, 2, 4, 8])
def test_lag_n_correct(lag):
    """lag_n at row i should equal y[i - n] (NaN for first n rows)."""
    df = _single_sku(n_weeks=15, y_start=1)
    feat = compute_features(df.copy())
    feat = feat.sort_values('week').reset_index(drop=True)

    col = f'lag_{lag}'
    for i in range(len(feat)):
        expected = float(feat['y'].iloc[i - lag]) if i >= lag else float('nan')
        actual = feat[col].iloc[i]
        if np.isnan(expected):
            assert np.isnan(actual), f'{col} at row {i}: expected NaN, got {actual}'
        else:
            assert actual == pytest.approx(expected), f'{col} at row {i}: expected {expected}, got {actual}'


# ── Rolling excludes current week ────────────────────────────────────────────

def test_rolling_excludes_current_week():
    """rolling_4_mean at week t should use t-1 through t-4, not t."""
    df = _single_sku(n_weeks=10, y_start=1)
    feat = compute_features(df.copy())
    feat = feat.sort_values('week').reset_index(drop=True)

    # At index 4 (y=5): shift(1).rolling(4) should use y[3], y[2], y[1], y[0] = 4, 3, 2, 1
    expected_mean = (4.0 + 3.0 + 2.0 + 1.0) / 4
    assert feat['rolling_4_mean'].iloc[4] == pytest.approx(expected_mean)

    # At index 5 (y=6): uses y[4], y[3], y[2], y[1] = 5, 4, 3, 2
    expected_mean2 = (5.0 + 4.0 + 3.0 + 2.0) / 4
    assert feat['rolling_4_mean'].iloc[5] == pytest.approx(expected_mean2)


# ── sell_price ffill-only (regression for bfill) ─────────────────────────────

def test_sell_price_no_bfill():
    """Pre-launch NaN sell_price must not be backfilled from future prices."""
    weeks = pd.date_range('2014-01-04', periods=6, freq='7D').strftime('%Y-%m-%d').tolist()
    df = pd.DataFrame({
        'week': weeks,
        'unique_id': 'A',
        'y': [0.0, 0.0, 1.0, 2.0, 1.0, 3.0],
        'snap': 0, 'has_event': 0, 'event_type_enc': 0,
        # sell_price is NaN for first 2 weeks (pre-launch), then has values
        'sell_price': [float('nan'), float('nan'), 2.5, 2.5, 3.0, 3.0],
        'dept_mean_sales': 5.0, 'cat_mean_sales': 10.0,
    })
    feat = compute_features(df.copy())
    feat = feat.sort_values('week').reset_index(drop=True)

    # First two rows must still be NaN (no backward fill from week 3)
    assert np.isnan(feat['sell_price'].iloc[0]), 'sell_price row 0 should be NaN (pre-launch)'
    assert np.isnan(feat['sell_price'].iloc[1]), 'sell_price row 1 should be NaN (pre-launch)'
    # Week 3 onward should be filled
    assert feat['sell_price'].iloc[2] == pytest.approx(2.5)


# ── price_change_pct correctness ─────────────────────────────────────────────

def test_price_change_pct():
    """price_change_pct should be pct_change after ffill, NaN→0, clipped [-1, 2]."""
    weeks = pd.date_range('2014-01-04', periods=5, freq='7D').strftime('%Y-%m-%d').tolist()
    df = pd.DataFrame({
        'week': weeks, 'unique_id': 'A', 'y': [1.0] * 5,
        'snap': 0, 'has_event': 0, 'event_type_enc': 0,
        'sell_price': [float('nan'), float('nan'), 2.0, 2.5, 2.0],
        'dept_mean_sales': 5.0, 'cat_mean_sales': 10.0,
    })
    feat = compute_features(df.copy())
    feat = feat.sort_values('week').reset_index(drop=True)

    # Pre-launch NaN price → pct_change is NaN → filled to 0
    assert feat['price_change_pct'].iloc[0] == pytest.approx(0.0)
    assert feat['price_change_pct'].iloc[1] == pytest.approx(0.0)
    assert feat['price_change_pct'].iloc[2] == pytest.approx(0.0)  # NaN prev → 0
    # 2.0 → 2.5: +25%
    assert feat['price_change_pct'].iloc[3] == pytest.approx(0.25, abs=1e-4)
    # 2.5 → 2.0: -20%
    assert feat['price_change_pct'].iloc[4] == pytest.approx(-0.20, abs=1e-4)


# ── Future-invariance ────────────────────────────────────────────────────────

def test_future_invariance():
    """Feature values for earlier weeks must not change when future weeks are appended."""
    df_short = _single_sku(n_weeks=10)
    extra_weeks = pd.date_range('2014-03-15', periods=5, freq='7D').strftime('%Y-%m-%d').tolist()
    extra = pd.DataFrame({
        'week': extra_weeks, 'unique_id': 'A',
        'y': [99.0] * 5, 'snap': 0, 'has_event': 0, 'event_type_enc': 0,
        'sell_price': 5.0, 'dept_mean_sales': 5.0, 'cat_mean_sales': 10.0,
    })
    df_long = pd.concat([df_short, extra], ignore_index=True)

    feat_short = compute_features(df_short.copy()).set_index('week')
    feat_long = compute_features(df_long.copy()).set_index('week')

    # Lag and rolling features for the first 10 weeks must be identical
    lag_rolling_cols = [c for c in FEATURE_COLS if c.startswith(('lag_', 'rolling_'))]
    common_weeks = feat_short.index.tolist()
    for col in lag_rolling_cols:
        short_vals = feat_short.loc[common_weeks, col].values
        long_vals = feat_long.loc[common_weeks, col].values
        np.testing.assert_array_equal(
            np.isnan(short_vals), np.isnan(long_vals),
            err_msg=f'{col}: NaN pattern differs when future weeks added',
        )
        mask = ~np.isnan(short_vals)
        if mask.any():
            np.testing.assert_allclose(
                short_vals[mask], long_vals[mask],
                rtol=1e-9,
                err_msg=f'{col}: values differ when future weeks added',
            )
