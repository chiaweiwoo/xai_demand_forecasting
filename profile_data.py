"""
Quick profiling of the raw M5 CSVs — no SQLite, no datasetsforecast parsing.
Reads the cached files in data/m5/datasets/ directly.

Usage:
    uv run python profile_data.py
"""

import pandas as pd

DATA_DIR  = 'data/m5/datasets'
STORE     = 'CA_1'
SNAP_COL  = 'snap_CA'


def sep(title: str) -> None:
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print('=' * 60)


def main() -> None:
    # ── 1. Sales ─────────────────────────────────────────────────
    sep('sales_train_evaluation.csv')
    sales_raw = pd.read_csv(f'{DATA_DIR}/sales_train_evaluation.csv')
    print(f'Shape: {sales_raw.shape}')
    print(f'Columns (first 10): {list(sales_raw.columns[:10])}')
    print(f'Unique stores: {sales_raw["store_id"].unique().tolist()}')
    print(f'Unique cats:   {sales_raw["cat_id"].unique().tolist()}')
    print(f'Unique depts:  {sales_raw["dept_id"].unique().tolist()}')

    store_sales = sales_raw[sales_raw['store_id'] == STORE].copy()
    print(f'\n{STORE} SKUs: {len(store_sales)}')

    # Daily columns
    day_cols = [c for c in store_sales.columns if c.startswith('d_')]
    print(f'Day columns: {len(day_cols)}  ({day_cols[0]} -> {day_cols[-1]})')

    # Melt to long
    id_cols = [c for c in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] if c in store_sales.columns]
    long = store_sales.melt(id_vars=id_cols, value_vars=day_cols, var_name='d', value_name='sales')
    print(f'\nLong format: {len(long):,} rows')
    print(f'Sales stats:\n{long["sales"].describe().round(2).to_string()}')
    print(f'\nZero-sales days: {(long["sales"] == 0).sum():,} ({(long["sales"] == 0).mean()*100:.1f}%)')

    # Per-SKU mean sales profile
    sep('Per-SKU mean sales (potential for target encoding)')
    sku_profile = long.groupby(['item_id', 'dept_id', 'cat_id'])['sales'].agg(
        mean='mean', median='median', std='std', zeros=lambda x: (x == 0).mean()
    ).reset_index()
    print(f'SKU count: {len(sku_profile)}')
    print(f'\nMean sales by category:')
    print(sku_profile.groupby('cat_id')['mean'].describe().round(2).to_string())
    print(f'\nMean sales by dept:')
    print(sku_profile.groupby('dept_id')['mean'].describe().round(2).to_string())
    print(f'\nSample of high vs low sellers:')
    print(sku_profile.sort_values('mean', ascending=False).head(5)[['item_id', 'dept_id', 'mean', 'zeros']].to_string(index=False))
    print('...')
    print(sku_profile.sort_values('mean').head(5)[['item_id', 'dept_id', 'mean', 'zeros']].to_string(index=False))

    # ── 2. Calendar ───────────────────────────────────────────────
    sep('calendar.csv')
    cal = pd.read_csv(f'{DATA_DIR}/calendar.csv')
    print(f'Shape: {cal.shape}')
    print(f'Columns: {list(cal.columns)}')
    print(f'\nDate range: {cal["date"].min()} -> {cal["date"].max()}')
    print(f'Total days: {len(cal)}')
    print(f'\nsnap_CA:')
    print(cal[SNAP_COL].value_counts().to_string())
    print(f'\nevent_type_1 distribution:')
    print(cal['event_type_1'].value_counts(dropna=False).to_string())
    print(f'\nevent_name_1 (top 10):')
    print(cal['event_name_1'].value_counts(dropna=False).head(10).to_string())

    # Weekly aggregation check
    cal['date'] = pd.to_datetime(cal['date'])
    cal['week_start'] = cal['date'].dt.to_period('W').dt.start_time
    cal['dow'] = cal['date'].dt.day_name()
    print(f'\nDay-of-week distribution (confirms data is daily):')
    print(cal['dow'].value_counts().to_string())

    # ── 3. Prices ─────────────────────────────────────────────────
    sep('sell_prices.csv')
    prices = pd.read_csv(f'{DATA_DIR}/sell_prices.csv')
    print(f'Shape: {prices.shape}')
    print(f'Columns: {list(prices.columns)}')
    store_prices = prices[prices['store_id'] == STORE]
    print(f'\n{STORE} price rows: {len(store_prices):,}')
    print(f'Price stats:\n{store_prices["sell_price"].describe().round(4).to_string()}')
    print(f'\nNull prices: {store_prices["sell_price"].isna().sum():,}')
    print(f'\nSample of wm_yr_wk values: {sorted(store_prices["wm_yr_wk"].unique()[:5])} ...')

    # Check how wm_yr_wk maps to dates
    wk_map = cal[['wm_yr_wk', 'date']].dropna(subset=['wm_yr_wk']).drop_duplicates('wm_yr_wk')
    print(f'\nwm_yr_wk -> date range (first 3 weeks):')
    for _, row in wk_map.head(3).iterrows():
        print(f'  {int(row["wm_yr_wk"])} -> {row["date"]}')

    # ── 4. Missingness / coverage ─────────────────────────────────
    sep('Coverage check: are prices present for all SKU-weeks?')
    sales_weeks = store_sales[['item_id']].copy()
    sales_weeks['n_weeks'] = len(day_cols) // 7  # approx
    price_coverage = store_prices.groupby('item_id')['wm_yr_wk'].count().rename('price_weeks')
    merged = sales_weeks.merge(price_coverage, on='item_id', how='left')
    print(f'SKUs with price data: {merged["price_weeks"].notna().sum()} / {len(merged)}')
    print(f'SKUs missing all prices: {merged["price_weeks"].isna().sum()}')
    print(f'Price weeks per SKU: min={merged["price_weeks"].min():.0f}  max={merged["price_weeks"].max():.0f}  mean={merged["price_weeks"].mean():.0f}')


if __name__ == '__main__':
    main()
