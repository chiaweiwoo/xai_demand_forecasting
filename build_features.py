"""
One-time feature store build. Run after ingest.py, before backtest.py.

Loads all raw data in a single pass, computes features once for every
(unique_id, week), and writes to the features table. Backtest and smoke_test
then do a plain SQL SELECT instead of recomputing features at each iteration.

Usage:
    uv run python build_features.py

Takes ~1 min. Always clears and rebuilds — safe to re-run.
"""

import time
import pandas as pd
from xai_forecast.db import get_conn, get_all_weeks, load_raw_window
from xai_forecast.features import compute_features, FEATURE_COLS, EXTERNAL_SIGNAL_COLS

DB_PATH = 'db/forecasting.db'
STORE_COLS = ['unique_id', 'week', 'y'] + FEATURE_COLS


def main() -> None:
    t0 = time.perf_counter()
    conn = get_conn(DB_PATH)

    weeks = get_all_weeks(conn)
    if not weeks:
        print('No data -- run: uv run python ingest.py')
        conn.close()
        return

    n_existing = conn.execute('SELECT COUNT(*) FROM features').fetchone()[0]
    if n_existing > 0:
        print(f'Clearing {n_existing:,} existing rows...')
        conn.execute('DELETE FROM features')
        conn.commit()

    print(f'Building feature store: {len(weeks)} weeks x 3049 SKUs...')

    # Load ALL raw data in one query (empty string sorts before any date)
    t = time.perf_counter()
    raw_df = load_raw_window(conn, '', weeks[-1])
    print(f'  Raw load:  {len(raw_df):,} rows  ({time.perf_counter()-t:.1f}s)')

    # Load external signals (populated by ingest_external.py)
    n_ext = conn.execute('SELECT COUNT(*) FROM external_signals').fetchone()[0]
    if n_ext == 0:
        print('  WARNING: external_signals table is empty. Run: uv run python ingest_external.py')
        ext_df = None
    else:
        ext_df = pd.read_sql(
            f"SELECT week, {', '.join(EXTERNAL_SIGNAL_COLS)} FROM external_signals",
            conn,
        )
        print(f'  External:  {len(ext_df):,} weeks of signals loaded')

    # Compute features in a single pass over all 278 weeks
    t = time.perf_counter()
    features_df = compute_features(raw_df, ext_df=ext_df)
    print(f'  Compute:   {len(features_df):,} rows  ({time.perf_counter()-t:.1f}s)')

    # Write to SQLite
    t = time.perf_counter()
    features_df[STORE_COLS].to_sql(
        'features', conn,
        if_exists='append',
        index=False,
        chunksize=10_000,
    )
    conn.commit()
    print(f'  Write:     ({time.perf_counter()-t:.1f}s)')

    n_written = conn.execute('SELECT COUNT(*) FROM features').fetchone()[0]
    conn.close()

    print(f'\nDone. {n_written:,} rows in features table  (total {time.perf_counter()-t0:.1f}s)')
    print('Next: uv run python backtest.py')


if __name__ == '__main__':
    main()
