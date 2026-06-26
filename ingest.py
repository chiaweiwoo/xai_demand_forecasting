"""
One-time data ingestion. Run this before backtest.py.

Downloads M5 data via datasetsforecast, builds the full weekly feature
matrix, and writes it to the `features` table in SQLite.

Usage:
    uv run python ingest.py
"""

import pandas as pd
from xai_forecast.features import load_and_prepare, FEATURE_COLS
from xai_forecast.db import init_db, get_conn, insert_features, get_weeks

DB_PATH = 'db/forecasting.db'


def main() -> None:
    print('Loading and preparing M5 data...')
    df = load_and_prepare('data')

    # Store week as ISO date string for clean SQL comparison
    df['week'] = df['week'].dt.strftime('%Y-%m-%d')

    keep_cols = ['week', 'unique_id', 'y'] + FEATURE_COLS
    df = df[keep_cols]

    print(f'  {df["unique_id"].nunique()} items | {df["week"].nunique()} weeks | {len(df):,} rows')
    print(f'  Sample row:\n{df.iloc[200].to_string()}\n')

    init_db(DB_PATH)
    conn = get_conn(DB_PATH)

    existing = get_weeks(conn)
    if existing:
        print(f'features table already has {len(existing)} weeks — skipping ingest.')
        print('Delete db/forecasting.db to re-ingest.')
        conn.close()
        return

    print('Writing to SQLite...')
    insert_features(conn, df)
    weeks = get_weeks(conn)
    print(f'Done. {len(weeks)} weeks written to {DB_PATH}')
    conn.close()


if __name__ == '__main__':
    main()
