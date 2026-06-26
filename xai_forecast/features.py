import pandas as pd
import numpy as np

STORE = 'CA_1'
SNAP_STATE = 'CA'

LAG_WEEKS = [1, 2, 4, 8, 52]
ROLL_WINDOWS = [4, 8, 13]

FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_52',
    'rolling_4_mean', 'rolling_8_mean', 'rolling_13_mean', 'rolling_4_std',
    'week_of_year', 'month', 'year',
    'snap', 'has_event', 'event_type_enc',
    'sell_price', 'price_change_pct',
    'dept_enc', 'cat_enc',
]

EVENT_TYPE_MAP = {'National': 1, 'Cultural': 2, 'Religious': 3, 'Sporting': 4}


def load_and_prepare(data_dir: str = 'data') -> pd.DataFrame:
    from datasetsforecast.m5 import M5
    Y_df, X_df, S_df = M5.load(directory=data_dir)

    mask = Y_df['unique_id'].str.endswith(f'_{STORE}')
    Y = Y_df[mask].copy()
    Y['ds'] = pd.to_datetime(Y['ds'])
    Y['week'] = Y['ds'].dt.to_period('W').dt.start_time

    Y_weekly = (
        Y.groupby(['unique_id', 'week'])['y']
        .sum()
        .reset_index()
    )

    if X_df is not None and len(X_df) > 0:
        X = X_df[X_df['unique_id'].str.endswith(f'_{STORE}')].copy()
        X['ds'] = pd.to_datetime(X['ds'])
        X['week'] = X['ds'].dt.to_period('W').dt.start_time

        snap_col = f'snap_{SNAP_STATE}'
        agg: dict = {}
        if 'sell_price' in X.columns:
            agg['sell_price'] = 'mean'
        if snap_col in X.columns:
            agg[snap_col] = 'max'
        if 'event_type_1' in X.columns:
            agg['event_type_1'] = lambda s: s.dropna().iloc[0] if s.notna().any() else None

        if agg:
            X_w = X.groupby(['unique_id', 'week']).agg(agg).reset_index()
            Y_weekly = Y_weekly.merge(X_w, on=['unique_id', 'week'], how='left')
            if snap_col in Y_weekly.columns:
                Y_weekly = Y_weekly.rename(columns={snap_col: '_snap_raw'})

    if S_df is not None and len(S_df) > 0:
        static_cols = [c for c in ['unique_id', 'dept_id', 'cat_id'] if c in S_df.columns]
        Y_weekly = Y_weekly.merge(S_df[static_cols], on='unique_id', how='left')

    df = Y_weekly.sort_values(['unique_id', 'week']).reset_index(drop=True)
    return _engineer(df)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in LAG_WEEKS:
        df[f'lag_{lag}'] = df.groupby('unique_id')['y'].transform(lambda x: x.shift(lag))

    for w in ROLL_WINDOWS:
        df[f'rolling_{w}_mean'] = df.groupby('unique_id')['y'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
    df['rolling_4_std'] = df.groupby('unique_id')['y'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).std().fillna(0)
    )

    df['week_of_year'] = df['week'].dt.isocalendar().week.astype(int)
    df['month'] = df['week'].dt.month
    df['year'] = df['week'].dt.year

    df['snap'] = df['_snap_raw'].fillna(0).astype(int) if '_snap_raw' in df.columns else 0

    if 'event_type_1' in df.columns:
        df['has_event'] = df['event_type_1'].notna().astype(int)
        df['event_type_enc'] = df['event_type_1'].map(EVENT_TYPE_MAP).fillna(0).astype(int)
    else:
        df['has_event'] = 0
        df['event_type_enc'] = 0

    if 'sell_price' in df.columns:
        df['sell_price'] = df.groupby('unique_id')['sell_price'].transform(
            lambda x: x.fillna(x.median())
        )
        df['price_change_pct'] = df.groupby('unique_id')['sell_price'].transform(
            lambda x: x.pct_change().fillna(0).clip(-1, 2)
        )
    else:
        df['sell_price'] = 0.0
        df['price_change_pct'] = 0.0

    df['dept_enc'] = df['dept_id'].astype('category').cat.codes.astype(int) if 'dept_id' in df.columns else 0
    df['cat_enc'] = df['cat_id'].astype('category').cat.codes.astype(int) if 'cat_id' in df.columns else 0

    return df
