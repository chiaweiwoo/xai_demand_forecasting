"""
One-time raw data ingestion. Run once before build_features.py.

Loads M5 via datasetsforecast, stores raw weekly data into SQLite.
No feature engineering here — fast and simple.

Usage:
    uv run python ingest.py
"""

import pandas as pd
from xai_forecast.db import get_conn, insert_raw

STORE = 'CA_1'
SNAP_STATE = 'CA'
DB_PATH = 'db/forecasting.db'
EVENT_TYPE_MAP = {'National': 1, 'Cultural': 2, 'Religious': 3, 'Sporting': 4}


def main() -> None:
    print('Loading M5 from datasetsforecast...')
    from datasetsforecast.m5 import M5
    Y_df, X_df, S_df = M5.load(directory='data')
    print(f'  Raw: {len(Y_df):,} daily rows loaded')

    # Filter to one store
    mask = Y_df['unique_id'].str.endswith(f'_{STORE}')
    Y = Y_df[mask].copy()
    Y['ds'] = pd.to_datetime(Y['ds'])
    Y['week'] = Y['ds'].dt.to_period('W').dt.start_time.dt.strftime('%Y-%m-%d')

    # weekly_sales: sum daily → weekly
    weekly_sales = (
        Y.groupby(['unique_id', 'week'])['y']
        .sum().reset_index()
    )
    print(f'  weekly_sales: {len(weekly_sales):,} rows | {weekly_sales["unique_id"].nunique()} items | {weekly_sales["week"].nunique()} weeks')

    # calendar: one row per week (snap + events — store-level, not item-level)
    snap_col = f'snap_{SNAP_STATE}'
    X = X_df[X_df['unique_id'].str.endswith(f'_{STORE}')].copy()
    X['ds'] = pd.to_datetime(X['ds'])
    X['week'] = X['ds'].dt.to_period('W').dt.start_time.dt.strftime('%Y-%m-%d')

    cal_agg: dict = {}
    if snap_col in X.columns:
        cal_agg[snap_col] = 'max'
    if 'event_type_1' in X.columns:
        cal_agg['event_type_1'] = lambda s: s.dropna().iloc[0] if s.notna().any() else None

    # Calendar is store-level so deduplicate across items
    X_week = X.drop_duplicates('week').set_index('week')
    calendar_rows = []
    for week in weekly_sales['week'].unique():
        row = {'week': week, 'snap': 0, 'has_event': 0, 'event_type_enc': 0}
        if week in X_week.index:
            r = X_week.loc[week]
            row['snap'] = int(r[snap_col]) if snap_col in X_week.columns and pd.notna(r.get(snap_col)) else 0
            et = r.get('event_type_1') if 'event_type_1' in X_week.columns else None
            row['has_event'] = 1 if pd.notna(et) else 0
            row['event_type_enc'] = EVENT_TYPE_MAP.get(et, 0) if pd.notna(et) else 0
        calendar_rows.append(row)
    calendar = pd.DataFrame(calendar_rows)
    print(f'  calendar: {len(calendar)} weeks | {calendar["has_event"].sum()} event weeks | {calendar["snap"].sum()} SNAP weeks')

    # prices: item × week
    if 'sell_price' in X_df.columns:
        Xp = X_df[X_df['unique_id'].str.endswith(f'_{STORE}')].copy()
        Xp['ds'] = pd.to_datetime(Xp['ds'])
        Xp['week'] = Xp['ds'].dt.to_period('W').dt.start_time.dt.strftime('%Y-%m-%d')
        prices = (
            Xp.groupby(['unique_id', 'week'])['sell_price']
            .mean().reset_index()
        )
    else:
        prices = pd.DataFrame(columns=['unique_id', 'week', 'sell_price'])
    print(f'  prices: {len(prices):,} rows')

    # item_meta: static per item
    if S_df is not None and len(S_df) > 0:
        meta = S_df[S_df['unique_id'].str.endswith(f'_{STORE}')].copy()
        meta['dept_enc'] = meta['dept_id'].astype('category').cat.codes.astype(int) if 'dept_id' in meta.columns else 0
        meta['cat_enc']  = meta['cat_id'].astype('category').cat.codes.astype(int)  if 'cat_id'  in meta.columns else 0
        item_meta = meta[['unique_id', 'dept_enc', 'cat_enc']]
    else:
        item_meta = pd.DataFrame({'unique_id': weekly_sales['unique_id'].unique(), 'dept_enc': 0, 'cat_enc': 0})
    print(f'  item_meta: {len(item_meta)} items')

    print('\nWriting to SQLite...')
    conn = get_conn(DB_PATH)

    # Guard: skip if already ingested
    cur = conn.execute('SELECT COUNT(*) FROM weekly_sales')
    if cur.fetchone()[0] > 0:
        print('weekly_sales already populated — skipping. Delete db/forecasting.db to re-ingest.')
        conn.close()
        return

    insert_raw(conn, weekly_sales, calendar, prices, item_meta)
    conn.close()

    n = len(weekly_sales)
    print(f'Done. {n:,} rows written to {DB_PATH}')


if __name__ == '__main__':
    main()
