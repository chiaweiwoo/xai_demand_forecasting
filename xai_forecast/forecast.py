import numpy as np
import pandas as pd
import lightgbm as lgb

from xai_forecast.features import FEATURE_COLS


def make_forecasts(
    model: lgb.LGBMRegressor,
    week_df: pd.DataFrame,
    forecast_week: str,
) -> pd.DataFrame:
    """
    Predict h=1 for a single week.
    week_df: rows already filtered to forecast_week from SQLite.
    Returns DataFrame with [unique_id, h1].
    """
    if week_df.empty:
        return pd.DataFrame(columns=['unique_id', 'h1'])
    X = week_df[FEATURE_COLS].fillna(0).values
    preds = model.predict(X).clip(min=0)
    return pd.DataFrame({'unique_id': week_df['unique_id'].values, 'h1': preds})
