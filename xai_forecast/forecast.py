import numpy as np
import pandas as pd
import lightgbm as lgb

from xai_forecast.features import FEATURE_COLS


def make_forecasts(
    model: lgb.LGBMRegressor,
    df: pd.DataFrame,
    forecast_weeks: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    For each of the 3 forecast horizons, predict using the feature row
    for that week (pre-built, uses actual lags — valid in retrospective backtest).
    Returns DataFrame with [unique_id, h1, h2, h3].
    """
    results: dict[str, dict] = {}

    for h, fw in enumerate(forecast_weeks[:3], start=1):
        fw_rows = df[df['week'] == fw][['unique_id'] + FEATURE_COLS]
        if fw_rows.empty:
            continue
        X = fw_rows[FEATURE_COLS].fillna(0).values
        preds = model.predict(X).clip(min=0)
        for uid, pred in zip(fw_rows['unique_id'], preds):
            if uid not in results:
                results[uid] = {'unique_id': uid, 'h1': np.nan, 'h2': np.nan, 'h3': np.nan}
            results[uid][f'h{h}'] = float(pred)

    return pd.DataFrame(list(results.values()))
