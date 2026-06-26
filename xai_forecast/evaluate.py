import numpy as np
import pandas as pd


def evaluate_h1(forecasts_df: pd.DataFrame, actuals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare h1 forecasts vs actuals for non-zero-actual SKUs.
    Returns [unique_id, actual, predicted, mae, mape].
    Zero-actual rows are excluded — MAPE is undefined there.
    Use week-level WMAPE (via flag_bad_weeks) for volume-weighted signal.
    """
    merged = forecasts_df[['unique_id', 'h1']].merge(
        actuals_df[['unique_id', 'y']], on='unique_id', how='inner'
    )
    merged = merged[merged['y'] > 0].copy()
    merged['actual']    = merged['y']
    merged['predicted'] = merged['h1']
    merged['mae']       = np.abs(merged['actual'] - merged['predicted'])
    merged['mape']      = merged['mae'] / merged['actual'] * 100
    return merged[['unique_id', 'actual', 'predicted', 'mae', 'mape']]


def flag_bad_weeks(
    all_evals: pd.DataFrame,
    window: int = 8,
    z_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Flag weeks where WMAPE spikes vs recent history.
    WMAPE = sum(|error|) / sum(actual) per week — volume-weighted, not dominated
    by near-zero-actual SKUs the way avg-MAPE is.

    all_evals must have columns: forecast_week, mae, actual.
    Returns week-level DataFrame: [forecast_week, wmape, zscore, is_bad_week].
    """
    wk = (
        all_evals.groupby('forecast_week')
        .agg(total_mae=('mae', 'sum'), total_actual=('actual', 'sum'))
        .reset_index()
        .sort_values('forecast_week')
    )
    wk['wmape']        = wk['total_mae'] / wk['total_actual'].clip(lower=1) * 100
    wk['rolling_mean'] = wk['wmape'].rolling(window, min_periods=3).mean()
    wk['rolling_std']  = wk['wmape'].rolling(window, min_periods=3).std().clip(lower=0.01)
    wk['zscore']       = (wk['wmape'] - wk['rolling_mean']) / wk['rolling_std']
    wk['is_bad_week']  = wk['zscore'] >= z_threshold
    return wk[['forecast_week', 'wmape', 'zscore', 'is_bad_week']]
