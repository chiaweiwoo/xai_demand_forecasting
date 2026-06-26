"""
Smoke test: one full train/forecast/evaluate/xai cycle from SQLite.
Run ingest.py first.

Usage:
    uv run python smoke_test.py
"""

import sys
import json
import time

from xai_forecast.db import (
    get_conn, get_weeks,
    load_features_window, load_features_week,
    insert_forecasts, insert_evaluations, insert_xai, week_summary,
)
from xai_forecast.train import train_model
from xai_forecast.forecast import make_forecasts
from xai_forecast.evaluate import evaluate_h1
from xai_forecast.xai import make_explainer, shap_payloads, counterfactual_payloads, contrastive_payloads
from xai_forecast.features import FEATURE_COLS

TRAIN_WINDOW = 156
TOP_N = 5
DB_PATH = 'db/forecasting.db'

_step_start = None


def step(label: str) -> None:
    global _step_start
    _step_start = time.perf_counter()
    print(f'\n[STEP] {label}')


def ok(label: str, detail: str = '') -> None:
    elapsed = time.perf_counter() - _step_start
    suffix = f' — {detail}' if detail else ''
    print(f'  [OK]   {label}{suffix}  ({elapsed:.2f}s)')


def fail(label: str, detail: str = '') -> None:
    elapsed = time.perf_counter() - _step_start
    suffix = f' — {detail}' if detail else ''
    print(f'  [FAIL] {label}{suffix}  ({elapsed:.2f}s)')
    sys.exit(1)


def check(label: str, condition: bool, detail: str = '') -> None:
    if condition:
        ok(label, detail)
    else:
        fail(label, detail)


def main() -> None:
    total_start = time.perf_counter()
    print('=' * 50)
    print('Smoke test')
    print('=' * 50)

    # ── 1. Connect ────────────────────────────────────
    step('Connect to SQLite')
    conn = get_conn(DB_PATH)
    weeks = get_weeks(conn)
    if not weeks:
        fail('No data found — run: uv run python ingest.py')
    check('Weeks in DB', len(weeks) > TRAIN_WINDOW, f'{len(weeks)} weeks')

    # ── 2. Load training window ───────────────────────
    step(f'Load training window (weeks 0–{TRAIN_WINDOW}) from SQLite')
    cutoff = weeks[TRAIN_WINDOW]
    window_start = weeks[0]
    train_df = load_features_window(conn, window_start, cutoff)
    check('Training rows loaded', len(train_df) > 0, f'{len(train_df):,} rows | cutoff={cutoff}')

    # ── 3. Train ──────────────────────────────────────
    step('Train LightGBM')
    model = train_model(train_df)
    check('Model trained', model is not None)
    check('Feature importances', len(model.feature_importances_) == len(FEATURE_COLS),
          f'{len(FEATURE_COLS)} features')

    # ── 4. Load forecast week ─────────────────────────
    forecast_week = weeks[TRAIN_WINDOW + 1]
    step(f'Load forecast week from SQLite ({forecast_week})')
    week_df = load_features_week(conn, forecast_week)
    check('Forecast week loaded', len(week_df) > 0, f'{len(week_df)} rows')

    # ── 5. Forecast ───────────────────────────────────
    step('Forecast h=1')
    fcst_df = make_forecasts(model, week_df, forecast_week)
    check('Forecasts returned', len(fcst_df) > 0, f'{len(fcst_df)} SKUs')
    check('No negative predictions', (fcst_df['h1'] >= 0).all())
    ok(f'Avg predicted={fcst_df["h1"].mean():.1f}  median={fcst_df["h1"].median():.1f}', '')

    # ── 6. Evaluate ───────────────────────────────────
    step('Evaluate h=1 vs actuals')
    actuals = week_df[['unique_id', 'y']]
    eval_df = evaluate_h1(fcst_df, actuals)
    eval_df['cutoff_week'] = cutoff
    check('Eval rows returned', len(eval_df) > 0, f'{len(eval_df)} rows')
    check('MAPE in valid range', eval_df['mape'].between(0, 1000).all())
    ok(f'Avg MAPE={eval_df["mape"].mean():.1f}%  median={eval_df["mape"].median():.1f}%', '')

    # ── 7. SHAP ───────────────────────────────────────
    top_items = eval_df.nlargest(TOP_N, 'mape')['unique_id'].tolist()
    actual_map = dict(zip(eval_df['unique_id'], eval_df['actual']))

    step(f'Build SHAP explainer + compute for top {TOP_N} items')
    explainer = make_explainer(model)
    shap_rows = shap_payloads(explainer, model, week_df, forecast_week, top_items, actual_map)
    check('SHAP rows', len(shap_rows) == TOP_N, f'{len(shap_rows)} rows')
    check('SHAP payload valid JSON', all(json.loads(r['payload']) for r in shap_rows))

    # ── 8. Counterfactual ─────────────────────────────
    step(f'Counterfactual for top {TOP_N} items')
    cf_rows = counterfactual_payloads(model, week_df, forecast_week, top_items, actual_map)
    check('Counterfactual rows', len(cf_rows) == TOP_N, f'{len(cf_rows)} rows')

    # ── 9. Contrastive ────────────────────────────────
    step(f'Contrastive for top {TOP_N} items')
    ct_rows = contrastive_payloads(explainer, week_df, forecast_week, top_items, eval_df, conn)
    check('Contrastive ran without error', True,
          f'{len(ct_rows)} rows (0 expected — no prior eval history at week {TRAIN_WINDOW})')

    # ── 10. SQLite write/read ─────────────────────────
    step('Write results to SQLite and read back')
    insert_forecasts(conn, [
        {'week_id': cutoff, 'item_id': r['unique_id'], 'h1': r['h1'], 'trained_at': 'smoke-test'}
        for r in fcst_df.to_dict('records')
    ])
    insert_evaluations(conn, [
        {'week_id': cutoff, 'item_id': r['unique_id'],
         'h1_mape': r['mape'], 'h1_mae': r['mae'], 'is_bad_week': 1, 'mape_zscore': 2.0}
        for r in eval_df.to_dict('records')
    ])
    insert_xai(conn, shap_rows + cf_rows)
    summary = week_summary(conn)
    conn.close()
    check('Summary readable', len(summary) == 1,
          f'{int(summary["n_items"].iloc[0])} items | avg MAPE {summary["avg_mape"].iloc[0]:.1f}%')

    # ── Done ──────────────────────────────────────────
    total = time.perf_counter() - total_start
    print(f'\n{"=" * 50}')
    print(f'All checks passed  (total {total:.1f}s)')
    print('=' * 50)


if __name__ == '__main__':
    main()
