"""
Stage 1 validation report for external_signals table.

Runs explicit anchor checks and implicit structural checks.
Exits 1 if any FAIL; exits 0 if all pass.

Usage:
    uv run python validate_external.py
"""

import sys

import pandas as pd

from xai_forecast.db import get_conn

DB_PATH = 'db/forecasting.db'

_PASS = '\033[32m[PASS]\033[0m'
_FAIL = '\033[31m[FAIL]\033[0m'
_WARN = '\033[33m[WARN]\033[0m'


def _check(label: str, ok: bool, detail: str = '') -> bool:
    status = _PASS if ok else _FAIL
    msg = f'  {status}  {label}'
    if detail:
        msg += f'  ({detail})'
    print(msg)
    return ok


def main() -> None:
    conn = get_conn(DB_PATH)
    df = pd.read_sql('SELECT * FROM external_signals ORDER BY week', conn)
    df['week_dt'] = pd.to_datetime(df['week'])
    conn.close()

    all_ok = True

    print('\nExternal signals validation\n' + '-' * 60)

    # ── Coverage ──────────────────────────────────────────────────
    print('\nCoverage:')

    n = len(df)
    all_ok &= _check('Row count == 278 fiscal weeks', n == 278, f'{n} rows')

    all_ok &= _check(
        'First week == 2011-01-29',
        df['week'].min() == '2011-01-29',
        df['week'].min(),
    )
    all_ok &= _check(
        'Last week == 2016-05-21',
        df['week'].max() == '2016-05-21',
        df['week'].max(),
    )

    for col in ['temp_mean', 'temp_max', 'temp_min', 'precip', 'heat_days', 'gas_price', 'consumer_sentiment']:
        n_null = df[col].isna().sum()
        all_ok &= _check(f'No nulls in {col}', n_null == 0, f'{n_null} nulls')

    # ── Implicit structural checks ────────────────────────────────
    print('\nImplicit checks:')

    n_bad_order = (df['temp_max'] < df['temp_mean']).sum() + (df['temp_mean'] < df['temp_min']).sum()
    all_ok &= _check('temp_max >= temp_mean >= temp_min (every week)', n_bad_order == 0,
                     f'{n_bad_order} violations')

    n_neg_precip = (df['precip'] < 0).sum()
    all_ok &= _check('precip >= 0', n_neg_precip == 0, f'{n_neg_precip} negative rows')

    n_bad_hdays = ((df['heat_days'] < 0) | (df['heat_days'] > 7)).sum()
    all_ok &= _check('heat_days in [0, 7]', n_bad_hdays == 0, f'{n_bad_hdays} out-of-range rows')

    gas_ok = ((df['gas_price'] >= 2.0) & (df['gas_price'] <= 5.0)).all()
    all_ok &= _check('gas_price in [2.0, 5.0]', gas_ok,
                     f'min={df.gas_price.min():.3f} max={df.gas_price.max():.3f}')

    tmean_ok = ((df['temp_mean'] >= 5) & (df['temp_mean'] <= 35)).all()
    all_ok &= _check('LA temp_mean in [5, 35] C', tmean_ok,
                     f'min={df.temp_mean.min():.1f} max={df.temp_mean.max():.1f}')

    sent_ok = ((df['consumer_sentiment'] >= 50) & (df['consumer_sentiment'] <= 110)).all()
    all_ok &= _check('consumer_sentiment in [50, 110]', sent_ok,
                     f'min={df.consumer_sentiment.min():.1f} max={df.consumer_sentiment.max():.1f}')

    # Seasonality: summer (Jun-Aug) temp_mean should be meaningfully higher than winter (Dec-Feb)
    summer = df[df['week_dt'].dt.month.isin([6, 7, 8])]['temp_mean'].mean()
    winter = df[df['week_dt'].dt.month.isin([12, 1, 2])]['temp_mean'].mean()
    seasonal_ok = (summer - winter) >= 5
    all_ok &= _check(
        'Summer temp_mean > winter temp_mean (seasonality check)',
        seasonal_ok,
        f'summer={summer:.1f}C winter={winter:.1f}C diff={summer-winter:.1f}C',
    )

    # temp_mean not flat (not a constant; check std > 1)
    tmean_std = df['temp_mean'].std()
    all_ok &= _check('temp_mean has variation (std > 1 C)', tmean_std > 1,
                     f'std={tmean_std:.2f}')

    # ── Explicit anchors ──────────────────────────────────────────
    print('\nExplicit anchor checks:')

    # Oct 2012 gas spike (CA refinery crisis) -- EIA peak was around 2012-10-08
    oct12 = df[(df['week_dt'].dt.year == 2012) & (df['week_dt'].dt.month == 10)]
    peak12 = oct12['gas_price'].max() if not oct12.empty else None
    if peak12 is not None:
        all_ok &= _check(
            'Oct 2012 gas spike >= 4.4 (CA refinery crisis)',
            peak12 >= 4.4,
            f'peak={peak12:.3f}',
        )
    else:
        all_ok &= _check('Oct 2012 gas rows present', False, 'no rows for Oct 2012')

    # Gas declined to <= 3.0 in early 2016
    q12016 = df[(df['week_dt'].dt.year == 2016) & (df['week_dt'].dt.month <= 3)]
    low16 = q12016['gas_price'].min() if not q12016.empty else None
    if low16 is not None:
        all_ok &= _check(
            'Early 2016 gas price <= 3.0 (oil crash decline)',
            low16 <= 3.0,
            f'min Q1-2016={low16:.3f}',
        )
    else:
        all_ok &= _check('Q1 2016 gas rows present', False, 'no rows for Q1 2016')

    # Aug 2011 sentiment drop (debt-ceiling crisis)
    aug11 = df[(df['week_dt'].dt.year == 2011) & (df['week_dt'].dt.month == 8)]
    sent11 = aug11['consumer_sentiment'].mean() if not aug11.empty else None
    if sent11 is not None:
        all_ok &= _check(
            'Aug 2011 consumer sentiment <= 62 (debt-ceiling crisis)',
            sent11 <= 62,
            f'avg={sent11:.1f}',
        )
    else:
        all_ok &= _check('Aug 2011 sentiment rows present', False, 'no Aug 2011 rows')

    # 2015 peak sentiment >= 90
    sent15 = df[df['week_dt'].dt.year == 2015]['consumer_sentiment'].max()
    all_ok &= _check(
        '2015 peak sentiment >= 90 (recovery)',
        sent15 >= 90,
        f'peak={sent15:.1f}',
    )

    # Drought check: 2012-2016 annual precip below 2011 (LA drought period)
    precip_2011 = df[df['week_dt'].dt.year == 2011]['precip'].sum()
    precip_drought = df[df['week_dt'].dt.year.isin([2013, 2014, 2015])]['precip'].sum() / 3
    drought_ok = precip_drought <= precip_2011 * 1.1
    all_ok &= _check(
        'Drought 2013-2015 avg precip <= 2011 precip (approx)',
        drought_ok,
        f'2011={precip_2011:.0f}mm  drought-avg={precip_drought:.0f}mm',
    )

    # ── Summary ───────────────────────────────────────────────────
    print('\n' + '-' * 60)
    if all_ok:
        print(f'{_PASS} All external signal checks passed. Stage 1 gate: OPEN.')
    else:
        print(f'{_FAIL} One or more checks failed. Do NOT proceed to Stage 2.')
        sys.exit(1)


if __name__ == '__main__':
    main()
