"""
Ingest pre-fetched external signals CSVs into the external_signals table.

Usage:
    uv run python ingest_external.py

Reads from external_data/ (committed CSVs, never hits the internet).
Writes to db/forecasting.db -> external_signals table.
Idempotent: clears the table at start, then rewrites.

Alignment logic:
  Weather (daily, LA) : aggregate Sat-Fri per fiscal week, stamp with the Saturday date.
  Gas (EIA, Monday-stamped weekly): each EIA row stamped on Monday; map it to the
      fiscal week (Sat start) it falls in, then forward-fill any uncovered weeks.
  Sentiment (monthly UMCSENT): forward-fill the monthly value to all weeks in that month.
"""

import sys
from pathlib import Path

import pandas as pd

from xai_forecast.db import get_conn

DB_PATH = 'db/forecasting.db'
EXT_DIR = Path('external_data')

# Fiscal week range (Saturday-stamped) — must match the M5 CA_1 window.
FIRST_WEEK = '2011-01-29'
LAST_WEEK  = '2016-05-21'


def _fiscal_weeks() -> pd.DatetimeIndex:
    """All Saturday fiscal-week start dates in the M5 window."""
    return pd.date_range(start=FIRST_WEEK, end=LAST_WEEK, freq='W-SAT')


# ── Weather ───────────────────────────────────────────────────────────────────

def _ingest_weather(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    wx = pd.read_csv(EXT_DIR / 'weather_raw.csv', parse_dates=['date'])

    results = []
    for sat in weeks:
        fri = sat + pd.Timedelta(days=6)
        mask = (wx['date'] >= sat) & (wx['date'] <= fri)
        week_rows = wx[mask]
        if week_rows.empty:
            results.append({
                'week': sat.strftime('%Y-%m-%d'),
                'temp_mean': None, 'temp_max': None, 'temp_min': None,
                'precip': None, 'heat_days': None,
            })
        else:
            results.append({
                'week':      sat.strftime('%Y-%m-%d'),
                'temp_mean': round(week_rows['temp_mean'].mean(), 2),
                'temp_max':  round(week_rows['temp_max'].max(), 2),
                'temp_min':  round(week_rows['temp_min'].min(), 2),
                'precip':    round(week_rows['precip'].sum(), 2),
                'heat_days': int((week_rows['temp_max'] > 32).sum()),
            })
    return pd.DataFrame(results)


# ── Gas price ─────────────────────────────────────────────────────────────────

def _ingest_gas(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    gas = pd.read_csv(EXT_DIR / 'gas_price_raw.csv', parse_dates=['date'])
    gas = gas.sort_values('date').reset_index(drop=True)

    # Map each EIA Monday-stamped row to the fiscal week it falls in.
    # A Monday falls into the fiscal week whose Saturday is <= Monday <= Friday.
    week_dates = pd.Series(weeks, name='week_start')

    def _find_fiscal_week(monday: pd.Timestamp) -> pd.Timestamp | None:
        # Saturday <= monday <= friday  =>  saturday = monday - weekday_offset
        # Monday weekday = 0; Saturday weekday = 5 (Mon=0, Sat=5)
        # Days from Monday back to the preceding Saturday: monday.dayofweek + 2 (since Sat=5 in iso, Mon=0 in pandas)
        # Actually: the fiscal week starts on Saturday. Monday is day 2 of that week (Sat=0,Sun=1,Mon=2,...,Fri=6).
        # So Saturday = Monday - 2 days.
        sat = monday - pd.Timedelta(days=2)
        # But only if that Saturday is in our week set
        sat_str = sat.strftime('%Y-%m-%d')
        if sat_str in week_dates.dt.strftime('%Y-%m-%d').values:
            return sat
        return None

    gas['fiscal_sat'] = gas['date'].apply(_find_fiscal_week)
    gas_mapped = gas.dropna(subset=['fiscal_sat']).copy()
    gas_mapped['week'] = gas_mapped['fiscal_sat'].dt.strftime('%Y-%m-%d')
    gas_mapped = gas_mapped[['week', 'gas_price']].drop_duplicates('week')

    # Build a full-week frame and forward-fill gaps
    full = pd.DataFrame({'week': weeks.strftime('%Y-%m-%d')})
    full = full.merge(gas_mapped, on='week', how='left')
    full['gas_price'] = full['gas_price'].ffill()
    return full


# ── Consumer sentiment ────────────────────────────────────────────────────────

def _ingest_sentiment(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    sent = pd.read_csv(EXT_DIR / 'consumer_sentiment_raw.csv', parse_dates=['date'])
    sent = sent.sort_values('date').reset_index(drop=True)

    # Forward-fill monthly value to all fiscal weeks in that month.
    full = pd.DataFrame({'week': weeks})
    full['year_month'] = full['week'].dt.to_period('M')
    sent['year_month'] = sent['date'].dt.to_period('M')

    sent_map = sent[['year_month', 'consumer_sentiment']].drop_duplicates('year_month')
    full = full.merge(sent_map, on='year_month', how='left')
    # Any remaining gaps (e.g. first partial month if survey not yet released): ffill
    full['consumer_sentiment'] = full['consumer_sentiment'].ffill()
    full['week'] = full['week'].dt.strftime('%Y-%m-%d')
    return full[['week', 'consumer_sentiment']]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    conn = get_conn(DB_PATH)
    weeks = _fiscal_weeks()
    print(f'\nIngesting external signals for {len(weeks)} fiscal weeks ({FIRST_WEEK} to {LAST_WEEK})')

    print('  Building weather aggregates (Sat-Fri) ...')
    wx_df = _ingest_weather(weeks)

    print('  Mapping gas prices to fiscal weeks ...')
    gas_df = _ingest_gas(weeks)

    print('  Forward-filling consumer sentiment ...')
    sent_df = _ingest_sentiment(weeks)

    # Merge all three on week
    df = wx_df.merge(gas_df, on='week', how='left').merge(sent_df, on='week', how='left')

    # Sanity check before writing
    n_weeks = len(df)
    n_null_gas  = df['gas_price'].isna().sum()
    n_null_sent = df['consumer_sentiment'].isna().sum()
    n_null_wx   = df['temp_mean'].isna().sum()
    if n_null_gas > 0 or n_null_sent > 0 or n_null_wx > 0:
        print(f'  WARNING: nulls in merged data — gas={n_null_gas}, sentiment={n_null_sent}, weather={n_null_wx}')

    print(f'  Writing {n_weeks} rows to external_signals ...')
    conn.execute('DELETE FROM external_signals')
    df.to_sql('external_signals', conn, if_exists='append', index=False)
    conn.commit()

    print(f'  Done. {n_weeks} rows written.')

    # Print a quick summary
    row = conn.execute('SELECT COUNT(*), MIN(week), MAX(week) FROM external_signals').fetchone()
    print(f'  Table: {row[0]} rows, {row[1]} to {row[2]}')
    conn.close()


if __name__ == '__main__':
    main()
