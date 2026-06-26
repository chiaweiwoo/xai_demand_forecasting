import numpy as np
import pandas as pd


def evaluate_h1(forecasts_df: pd.DataFrame, actuals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare h1 forecasts vs actuals.
    forecasts_df: [unique_id, h1, ...]
    actuals_df:   [unique_id, y]
    Returns [unique_id, actual, predicted, mae, mape].
    """
    merged = forecasts_df[['unique_id', 'h1']].merge(
        actuals_df[['unique_id', 'y']], on='unique_id', how='inner'
    )
    merged = merged[merged['y'] > 0].copy()
    merged['actual'] = merged['y']
    merged['predicted'] = merged['h1']
    merged['mae'] = np.abs(merged['actual'] - merged['predicted'])
    merged['mape'] = merged['mae'] / merged['actual'] * 100
    return merged[['unique_id', 'actual', 'predicted', 'mae', 'mape']]


def flag_bad_weeks(
    all_evals: pd.DataFrame,
    window: int = 8,
    z_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Compute rolling z-score of weekly avg MAPE.
    Returns week-level DataFrame with [cutoff_week, avg_mape, zscore, is_bad_week].
    """
    wk = (
        all_evals.groupby('cutoff_week')['mape']
        .mean()
        .reset_index()
        .rename(columns={'mape': 'avg_mape'})
        .sort_values('cutoff_week')
    )
    wk['rolling_mean'] = wk['avg_mape'].rolling(window, min_periods=3).mean()
    wk['rolling_std'] = wk['avg_mape'].rolling(window, min_periods=3).std().clip(lower=0.01)
    wk['zscore'] = (wk['avg_mape'] - wk['rolling_mean']) / wk['rolling_std']
    wk['is_bad_week'] = wk['zscore'] >= z_threshold
    return wk
