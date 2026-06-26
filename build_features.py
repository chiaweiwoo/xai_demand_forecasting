"""
Feature engineering. Run after ingest.py, before backtest.py.

Reads raw tables from SQLite, computes the feature matrix,
writes to the features table. Run once — safe to re-run (skips if done).

Leakage guarantees:
  - All lag features use shift(n): lag_1 at week t = sales[t-1]
  - All rolling features use shift(1) before rolling: no current-week data
  - sell_price NaN filled with ffill (last known price, not global median)
  - price_change_pct computed after ffill

Usage:
    uv run python build_features.py
"""

import warnings
import pandas as pd
from xai_forecast.db import get_conn, insert_features, get_weeks
from xai_forecast.features import FEATURE_COLS, LAG_WEEKS, ROLL_WINDOWS, EVENT_TYPE_MAP

DB_PATH = 'db/forecasting.db'

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def main() -> None:
    conn = get_conn(DB_PATH)

    # Guard: skip if features already built
    if get_weeks(conn):
        print('features table already populated — skipping.')
        print('Delete db/forecasting.db and re-run migrate.py + ingest.py to rebuild.')
        conn.close()
        return

    print('Reading raw tables from SQLite...')
    sales    = pd.read_sql('SELECT * FROM weekly_sales ORDER BY unique_id, week', conn)
    calendar = pd.read_sql('SELECT * FROM calendar ORDER BY week', conn)
    prices   = pd.read_sql('SELECT * FROM prices ORDER BY unique_id, week', conn)
    meta     = pd.read_sql('SELECT * FROM item_meta', conn)
    print(f'  sales: {len(sales):,} rows | calendar: {len(calendar)} weeks | prices: {len(prices):,} rows')

    print('Joining tables...')
    df = (
        sales
        .merge(calendar, on='week', how='left')
        .merge(prices,   on=['unique_id', 'week'], how='left')
        .merge(meta,     on='unique_id', how='left')
        .sort_values(['unique_id', 'week'])
        .reset_index(drop=True)
    )

    print('Engineering features...')

    # Lag features — purely backward-looking
    for lag in LAG_WEEKS:
        df[f'lag_{lag}'] = df.groupby('unique_id')['y'].transform(lambda x: x.shift(lag))

    # Rolling features — shift(1) before rolling to exclude current week
    for w in ROLL_WINDOWS:
        df[f'rolling_{w}_mean'] = df.groupby('unique_id')['y'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
    df['rolling_4_std'] = df.groupby('unique_id')['y'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).std().fillna(0)
    )

    # Calendar (already in df from calendar join)
    df['week_of_year'] = pd.to_datetime(df['week']).dt.isocalendar().week.astype(int)
    df['month']        = pd.to_datetime(df['week']).dt.month
    df['year']         = pd.to_datetime(df['week']).dt.year

    # Price — ffill within each item (last known price, no future leakage)
    df['sell_price'] = df.groupby('unique_id')['sell_price'].transform(
        lambda x: x.ffill().bfill()   # bfill only for items with no early price
    )
    df['price_change_pct'] = df.groupby('unique_id')['sell_price'].transform(
        lambda x: x.pct_change(fill_method=None).fillna(0).clip(-1, 2)
    )

    # Categorical encodings (already in df from meta join)
    df['dept_enc'] = df['dept_enc'].fillna(0).astype(int)
    df['cat_enc']  = df['cat_enc'].fillna(0).astype(int)
    df['snap']     = df['snap'].fillna(0).astype(int)
    df['has_event']      = df['has_event'].fillna(0).astype(int)
    df['event_type_enc'] = df['event_type_enc'].fillna(0).astype(int)

    keep = ['week', 'unique_id', 'y'] + FEATURE_COLS
    out = df[keep]

    print(f'  Output: {len(out):,} rows × {len(out.columns)} columns')
    print(f'  NaN in features: {out[FEATURE_COLS].isna().sum().sum()} '
          f'(expected: lag/rolling NaN for first ~52 weeks)')

    print('Writing features table to SQLite...')
    insert_features(conn, out)
    conn.close()

    weeks = get_weeks(get_conn(DB_PATH))
    print(f'Done. {len(weeks)} weeks in features table.')
    print('Next: uv run python smoke_test.py')


if __name__ == '__main__':
    main()
