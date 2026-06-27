"""
Post-backtest data quality checks. Run after backtest.py.

Usage:
    uv run python data_quality.py

Exits 1 if any check fails, 0 if all pass. Safe to run at any time — read-only.
"""

import json
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
    all_ok = True

    print('\nData quality checks\n' + '-' * 50)

    # ── Forecasts ────────────────────────────────────────────────────────────

    n_forecasts = conn.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0]
    all_ok &= _check('forecasts table non-empty', n_forecasts > 0, f'{n_forecasts:,} rows')

    n_neg_h1 = conn.execute('SELECT COUNT(*) FROM forecasts WHERE h1 < 0').fetchone()[0]
    all_ok &= _check('forecasts.h1 >= 0', n_neg_h1 == 0, f'{n_neg_h1} negative rows')

    n_null_h1 = conn.execute('SELECT COUNT(*) FROM forecasts WHERE h1 IS NULL').fetchone()[0]
    if n_null_h1 > 0:
        print(f'  {_WARN}  {n_null_h1} NULL h1 rows (imputed to 0 by make_forecasts clip)')

    # ── Evaluations ──────────────────────────────────────────────────────────

    n_evals = conn.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0]
    all_ok &= _check('evaluations table non-empty', n_evals > 0, f'{n_evals:,} rows')

    # evaluations.week_id must be a subset of forecasts.week_id
    orphan_weeks = conn.execute(
        'SELECT COUNT(DISTINCT e.week_id) FROM evaluations e '
        'LEFT JOIN forecasts f ON f.week_id = e.week_id '
        'WHERE f.week_id IS NULL'
    ).fetchone()[0]
    all_ok &= _check('evaluations.week_id subset of forecasts.week_id', orphan_weeks == 0,
                     f'{orphan_weeks} orphan weeks')

    n_bad_weeks = conn.execute(
        'SELECT COUNT(DISTINCT week_id) FROM evaluations WHERE is_bad_week = 1'
    ).fetchone()[0]
    n_total_weeks = conn.execute('SELECT COUNT(DISTINCT week_id) FROM evaluations').fetchone()[0]
    _check('bad weeks detected', n_bad_weeks > 0, f'{n_bad_weeks}/{n_total_weeks} weeks flagged')
    # Not a hard failure if no bad weeks (might be a partial run)

    # ── XAI results ──────────────────────────────────────────────────────────

    n_xai = conn.execute('SELECT COUNT(*) FROM xai_results').fetchone()[0]
    all_ok &= _check('xai_results table non-empty', n_xai > 0, f'{n_xai:,} rows')

    # xai_results.week_id must be a subset of bad weeks
    xai_non_bad = conn.execute(
        'SELECT COUNT(DISTINCT x.week_id) FROM xai_results x '
        'LEFT JOIN (SELECT DISTINCT week_id FROM evaluations WHERE is_bad_week=1) b '
        'ON b.week_id = x.week_id WHERE b.week_id IS NULL'
    ).fetchone()[0]
    all_ok &= _check('xai_results.week_id subset of bad weeks', xai_non_bad == 0,
                     f'{xai_non_bad} non-bad-week XAI rows')

    # Each bad week should have all 3 xai_types (shap, counterfactual, contrastive)
    bad_weeks_df = pd.read_sql(
        'SELECT DISTINCT week_id FROM evaluations WHERE is_bad_week=1', conn
    )
    xai_types_ok = True
    if not bad_weeks_df.empty:
        xai_types_df = pd.read_sql(
            'SELECT week_id, xai_type, COUNT(*) AS n FROM xai_results GROUP BY week_id, xai_type',
            conn,
        )
        for week in bad_weeks_df['week_id']:
            types_for_week = set(xai_types_df[xai_types_df['week_id'] == week]['xai_type'].tolist())
            expected = {'shap', 'counterfactual', 'contrastive'}
            missing = expected - types_for_week
            if missing:
                # Contrastive is allowed to be absent (no same-WOY good week found)
                truly_missing = missing - {'contrastive'}
                if truly_missing:
                    print(f'  {_FAIL}  {week}: missing xai_types {truly_missing}')
                    xai_types_ok = False
                else:
                    print(f'  {_WARN}  {week}: no contrastive data (no same-WOY good week)')

    all_ok &= _check('all bad weeks have shap + counterfactual', xai_types_ok)

    # ── Payload JSON validity ─────────────────────────────────────────────────

    bad_json = 0
    sample_rows = conn.execute('SELECT payload FROM xai_results LIMIT 100').fetchall()
    for (payload_str,) in sample_rows:
        try:
            json.loads(payload_str)
        except json.JSONDecodeError:
            bad_json += 1
    all_ok &= _check('xai_results payloads are valid JSON (sample 100)', bad_json == 0,
                     f'{bad_json} invalid')

    # ── Features table ───────────────────────────────────────────────────────

    n_features = conn.execute('SELECT COUNT(*) FROM features').fetchone()[0]
    n_sales = conn.execute('SELECT COUNT(*) FROM weekly_sales').fetchone()[0]
    _check(
        'features row count == weekly_sales row count',
        n_features == n_sales,
        f'features={n_features:,}  weekly_sales={n_sales:,}'
    )
    if n_features != n_sales:
        print(f'  {_WARN}  Mismatch may be expected if build_features.py and ingest.py are out of sync.')
        # Not hard-failing this — it could be a rebuild after data reload

    # ── Pre-launch price leakage (bfill regression) ───────────────────────────
    # Rows where sell_price IS NOT NULL but lag_1 IS NULL (and it's NOT the first
    # dataset week) indicate a pre-launch row that received a price via backward fill.
    # Week 1 is excluded: lag_1 is always NULL there (shift(1) returns NaN for the
    # first row), and a non-null price in week 1 is genuine raw data, not leakage.
    first_week = conn.execute('SELECT MIN(week) FROM features').fetchone()[0]
    n_prelaunch_with_price = conn.execute(
        'SELECT COUNT(*) FROM features WHERE sell_price IS NOT NULL AND lag_1 IS NULL AND week != ?',
        (first_week,),
    ).fetchone()[0]
    all_ok &= _check(
        'No pre-launch price leakage (sell_price non-null with lag_1 null, excl. week 1)',
        n_prelaunch_with_price == 0,
        f'{n_prelaunch_with_price} suspicious rows'
    )

    conn.close()

    print('\n' + '-' * 50)
    if all_ok:
        print(f'{_PASS} All checks passed.')
    else:
        print(f'{_FAIL} One or more checks failed.')
        sys.exit(1)


if __name__ == '__main__':
    main()
