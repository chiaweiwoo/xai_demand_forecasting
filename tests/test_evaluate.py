"""Group B — evaluation and bad-week flagging tests."""

import numpy as np
import pandas as pd
import pytest

from xai_forecast.evaluate import evaluate_h1, flag_bad_weeks


# ── evaluate_h1 ──────────────────────────────────────────────────────────────

def test_evaluate_h1_drops_zero_actual():
    forecasts = pd.DataFrame({'unique_id': ['A', 'B', 'C'], 'h1': [5.0, 3.0, 1.0]})
    actuals   = pd.DataFrame({'unique_id': ['A', 'B', 'C'], 'y':  [0.0, 4.0, 0.0]})
    result = evaluate_h1(forecasts, actuals)
    assert 'A' not in result['unique_id'].values
    assert 'C' not in result['unique_id'].values
    assert list(result['unique_id']) == ['B']


def test_evaluate_h1_mape_formula():
    forecasts = pd.DataFrame({'unique_id': ['A'], 'h1': [12.0]})
    actuals   = pd.DataFrame({'unique_id': ['A'], 'y':  [8.0]})
    result = evaluate_h1(forecasts, actuals)
    assert result['mae'].iloc[0] == pytest.approx(4.0)
    assert result['mape'].iloc[0] == pytest.approx(50.0)  # |12-8|/8*100


def test_evaluate_h1_all_zeros_returns_empty():
    forecasts = pd.DataFrame({'unique_id': ['A', 'B'], 'h1': [5.0, 3.0]})
    actuals   = pd.DataFrame({'unique_id': ['A', 'B'], 'y':  [0.0, 0.0]})
    result = evaluate_h1(forecasts, actuals)
    assert len(result) == 0


# ── flag_bad_weeks ────────────────────────────────────────────────────────────

def _make_evals(wmapes: list[float]) -> pd.DataFrame:
    """Build all_evals DataFrame from a list of per-week WMAPE values (as mae/actual=1)."""
    weeks = [f'2014-{i:02d}-01' for i in range(1, len(wmapes) + 1)]
    records = []
    for w, wmape in zip(weeks, wmapes):
        # mae=wmape, actual=1 → WMAPE = mae/actual*100 = wmape*100
        records.append({'forecast_week': w, 'unique_id': 'A', 'mae': wmape, 'actual': 1.0})
    return pd.DataFrame(records)


def test_flag_bad_weeks_wmape_formula():
    """WMAPE = Σ|mae| / Σactual * 100 per week."""
    evals = pd.DataFrame([
        {'forecast_week': '2014-01-01', 'unique_id': 'A', 'mae': 2.0, 'actual': 10.0},
        {'forecast_week': '2014-01-01', 'unique_id': 'B', 'mae': 3.0, 'actual': 10.0},
    ])
    result = flag_bad_weeks(evals)
    # WMAPE = (2+3)/(10+10)*100 = 25%
    assert result['wmape'].iloc[0] == pytest.approx(25.0)


def test_flag_bad_weeks_z_threshold():
    """A week whose WMAPE is ≥ 1.5 std devs above rolling mean should be flagged."""
    # 8 normal weeks (low WMAPE), then 1 spike
    low = [0.01] * 8
    spike = 1.0
    evals = _make_evals(low + [spike])
    result = flag_bad_weeks(evals).reset_index(drop=True)
    assert bool(result.iloc[-1]['is_bad_week']), 'Spike week should be flagged'
    assert not bool(result.iloc[0]['is_bad_week']), 'First week should not be flagged'


def test_flag_bad_weeks_early_weeks_nan_zscore():
    """First few weeks (< min_periods=3) must have NaN z-score — not imputed to 0."""
    evals = _make_evals([0.1, 0.1, 0.5, 0.1, 0.1])
    result = flag_bad_weeks(evals).reset_index(drop=True)
    # Weeks 0 and 1 have fewer than 3 points in rolling window → z-score should be NaN
    assert np.isnan(result['zscore'].iloc[0])
    assert np.isnan(result['zscore'].iloc[1])


def test_flag_bad_weeks_no_negative_mape():
    """flag_bad_weeks must not produce negative WMAPE (actuals clipped to >= 1)."""
    evals = _make_evals([0.0, 0.0, 0.0])
    result = flag_bad_weeks(evals)
    assert (result['wmape'] >= 0).all()
