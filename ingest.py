"""
One-time raw data ingestion from M5 CSVs into SQLite.
Reads directly from data/m5/datasets/ — no datasetsforecast parsing.

Week key = Saturday date string (Walmart fiscal week start), derived from wm_yr_wk.
All tables join on `week` (str) and `unique_id` (str).

Usage:
    uv run python ingest.py
"""

import pandas as pd
from xai_forecast.db import get_conn, insert_raw

DATA_DIR  = 'data/m5/datasets'
STORE     = 'CA_1'
SNAP_COL  = 'snap_CA'
DB_PATH   = 'db/forecasting.db'

EVENT_TYPE_MAP = {'National': 1, 'Cultural': 2, 'Religious': 3, 'Sporting': 4}


def main() -> None:
    # ── Calendar: build d_N -> wm_yr_wk -> week_start mapping ───────────────
    print('Reading calendar...')
    cal = pd.read_csv(f'{DATA_DIR}/calendar.csv', parse_dates=['date'])
    cal = cal.sort_values('date').reset_index(drop=True)

    # d_N index: d_1 = first row (2011-01-29, Saturday)
    cal['d'] = 'd_' + (cal.index + 1).astype(str)

    # week_start = earliest date in each fiscal week (the Saturday)
    week_starts = cal.groupby('wm_yr_wk')['date'].min().rename('week_start')
    cal = cal.merge(week_starts, on='wm_yr_wk')
    cal['week'] = cal['week_start'].dt.strftime('%Y-%m-%d')

    # d -> week lookup for sales aggregation
    d_to_week = cal.set_index('d')['week'].to_dict()

    # Calendar table: one row per fiscal week
    calendar = (
        cal.groupby('week').agg(
            snap       = (SNAP_COL,       'max'),
            has_event  = ('event_type_1', lambda x: int(x.notna().any())),
            event_type_enc = ('event_type_1',
                lambda x: EVENT_TYPE_MAP.get(x.dropna().iloc[0], 0) if x.notna().any() else 0),
        )
        .reset_index()
    )
    print(f'  calendar: {len(calendar)} weeks | '
          f'{calendar["has_event"].sum()} event weeks | {calendar["snap"].sum()} SNAP weeks')

    # ── Sales: melt wide -> long, aggregate daily -> weekly ─────────────────
    print('Reading sales...')
    sales_raw = pd.read_csv(f'{DATA_DIR}/sales_train_evaluation.csv')
    store_df  = sales_raw[sales_raw['store_id'] == STORE].copy()
    day_cols  = [c for c in store_df.columns if c.startswith('d_')]

    print(f'  {len(store_df)} SKUs x {len(day_cols)} days -> melting...')
    long = store_df.melt(
        id_vars=['item_id', 'dept_id', 'cat_id'],
        value_vars=day_cols,
        var_name='d', value_name='y',
    )
    long['week']      = long['d'].map(d_to_week)
    long['unique_id'] = long['item_id'] + '_' + STORE

    weekly_sales = (
        long.groupby(['unique_id', 'week'], observed=True)['y']
        .sum().reset_index()
    )
    print(f'  weekly_sales: {len(weekly_sales):,} rows | '
          f'{weekly_sales["unique_id"].nunique()} items | {weekly_sales["week"].nunique()} weeks')

    # ── Prices: wm_yr_wk -> week string, then weekly mean ───────────────────
    print('Reading prices...')
    wk_map    = cal[['wm_yr_wk', 'week']].drop_duplicates()
    prices_raw = pd.read_csv(f'{DATA_DIR}/sell_prices.csv')
    prices_raw = prices_raw[prices_raw['store_id'] == STORE].copy()
    prices_raw = prices_raw.merge(wk_map, on='wm_yr_wk')
    prices_raw['unique_id'] = prices_raw['item_id'] + '_' + STORE

    prices = (
        prices_raw.groupby(['unique_id', 'week'], observed=True)['sell_price']
        .mean().reset_index()
    )
    print(f'  prices: {len(prices):,} rows | null prices: {prices["sell_price"].isna().sum()}')

    # ── Item meta: dept/cat target encoding (mean weekly sales) ─────────────
    print('Building item meta...')
    meta = long[['unique_id', 'item_id', 'dept_id', 'cat_id']].drop_duplicates('unique_id')

    # Mean weekly sales per dept and cat — static prior, no temporal leakage
    sku_weekly_mean = (
        long.groupby(['unique_id', 'week'], observed=True)['y'].sum()
        .groupby('unique_id').mean()
        .rename('sku_mean')
    )
    dept_mean = (
        sku_weekly_mean.reset_index()
        .merge(meta[['unique_id', 'dept_id']], on='unique_id')
        .groupby('dept_id')['sku_mean'].mean()
        .rename('dept_mean_sales')
    )
    cat_mean = (
        sku_weekly_mean.reset_index()
        .merge(meta[['unique_id', 'cat_id']], on='unique_id')
        .groupby('cat_id')['sku_mean'].mean()
        .rename('cat_mean_sales')
    )

    meta = (
        meta
        .merge(dept_mean, on='dept_id')
        .merge(cat_mean,  on='cat_id')
    )
    item_meta = meta[['unique_id', 'dept_id', 'cat_id', 'dept_mean_sales', 'cat_mean_sales']]
    print(f'  item_meta: {len(item_meta)} items')
    print(f'  dept_mean_sales range: {item_meta["dept_mean_sales"].min():.2f} - {item_meta["dept_mean_sales"].max():.2f}')
    print(f'  cat_mean_sales range:  {item_meta["cat_mean_sales"].min():.2f} - {item_meta["cat_mean_sales"].max():.2f}')

    # ── Write to SQLite ──────────────────────────────────────────────────────
    print('\nWriting to SQLite...')
    conn = get_conn(DB_PATH)

    cur = conn.execute('SELECT COUNT(*) FROM weekly_sales')
    if cur.fetchone()[0] > 0:
        print('weekly_sales already populated — delete db/forecasting.db to re-ingest.')
        conn.close()
        return

    insert_raw(conn, weekly_sales, calendar, prices, item_meta)
    conn.close()
    print(f'Done. {len(weekly_sales):,} rows written to {DB_PATH}')


if __name__ == '__main__':
    main()
