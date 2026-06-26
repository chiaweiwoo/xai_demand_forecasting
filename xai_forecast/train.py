import pandas as pd
import lightgbm as lgb

from xai_forecast.features import FEATURE_COLS

_LGB_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': 42,
}


def train_model(df: pd.DataFrame) -> lgb.LGBMRegressor:
    train = df.dropna(subset=FEATURE_COLS + ['y'])
    X = train[FEATURE_COLS]
    y = train['y'].clip(lower=0)
    model = lgb.LGBMRegressor(**_LGB_PARAMS)
    model.fit(X, y)
    return model
