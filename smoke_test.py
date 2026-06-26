"""
Smoke test: one full train/forecast/evaluate/xai cycle.
Requires ingest.py to have run first. Schema is created automatically by get_conn().

Usage:
    uv run python smoke_test.py

Expected: ~30-60 sec (SQL + feature compute + train + XAI).
"""

import sys
import json
import time

from xai_forecast.db import (
    get_conn, get_all_weeks, load_raw_window,
    insert_forecasts, insert_evaluations, insert_xai, week_summary,
)
from xai_forecast.features import compute_features, FEATURE_COLS, HISTORY_BUFFER
from xai_forecast.train import train_model
from xai_forecast.forecast import make_forecasts
from xai_forecast.evaluate import evaluate_h1
from xai_forecast.xai import make_explainer, shap_payloads, counterfactual_payloads, contrastive_payloads

TRAIN_WINDOW = 156
TOP_N = 5
DB_PATH = 'db/forecasting.db'

_t0 = None


def step(label: str) -> None:
    global _t0
    _t0 = time.perf_counter()
    print(f'\n[STEP] {label}')


def ok(label: str, detail: str = '') -> None:
    print(f'  [OK]   {label}' + (f' -- {detail}' if detail else '') + f'  ({time.perf_counter()-_t0:.2f}s)')


def fail(label: str, detail: str = '') -> None:
    print(f'  [FAIL] {label}' + (f' -- {detail}' if detail else '') + f'  ({time.perf_counter()-_t0:.2f}s)')
    sys.exit(1)


def check(label: str, condition: bool, detail: str = '') -> None:
    ok(label, detail) if condition else fail(label, detail)


def main() -> None:
    total = time.perf_counter()
    print('=' * 50)
    print('Smoke test')
    print('=' * 50)

    step('Connect + get weeks from SQLite')
    conn  = get_conn(DB_PATH)
    weeks = get_all_weeks(conn)
    if not weeks:
        fail('No data -- run: uv run python ingest.py')
    check('Weeks in DB', len(weeks) > TRAIN_WINDOW, f'{len(weeks)} weeks')

    step('Load raw window + compute features (train window)')
    cutoff       = weeks[TRAIN_WINDOW]
    forecast_week = weeks[TRAIN_WINDOW + 1]
    window_start = weeks[0]
    buffer_start = weeks[max(0, TRAIN_WINDOW - TRAIN_WINDOW - HISTORY_BUFFER)]  # weeks[0]
    raw_df       = load_raw_window(conn, buffer_start, cutoff)
    features_df  = compute_features(raw_df)
    train_df     = features_df[features_df['week'] > window_start].dropna(subset=FEATURE_COLS)
    check('Training rows', len(train_df) > 0, f'{len(train_df):,} rows | cutoff={cutoff}')
    missing = [c for c in FEATURE_COLS if c not in train_df.columns]
    check('All feature cols present', not missing, f'Missing: {missing}' if missing else '')

    step('Train LightGBM')
    model = train_model(train_df)
    check('Model trained', model is not None)
    check('Feature importances', len(model.feature_importances_) == len(FEATURE_COLS))

    step('Load + compute features for forecast week')
    buf_start = weeks[max(0, TRAIN_WINDOW + 1 - HISTORY_BUFFER)]
    raw_fcst  = load_raw_window(conn, buf_start, forecast_week)
    feat_fcst = compute_features(raw_fcst)
    week_df   = feat_fcst[feat_fcst['week'] == forecast_week]
    check('Forecast week rows', len(week_df) > 0, f'{len(week_df)} SKUs | week={forecast_week}')

    step('Forecast h=1')
    fcst_df = make_forecasts(model, week_df, forecast_week)
    check('Forecasts returned', len(fcst_df) > 0, f'{len(fcst_df)} SKUs')
    check('No negatives', (fcst_df['h1'] >= 0).all())
    ok(f'Avg predicted={fcst_df["h1"].mean():.1f}  median={fcst_df["h1"].median():.1f}')

    step('Evaluate h=1 vs actuals')
    eval_df = evaluate_h1(fcst_df, week_df[['unique_id', 'y']])
    eval_df['forecast_week'] = forecast_week
    check('Eval rows', len(eval_df) > 0, f'{len(eval_df)} rows')
    check('MAPE non-negative', (eval_df['mape'] >= 0).all())
    ok(f'Avg MAPE={eval_df["mape"].mean():.1f}%  median={eval_df["mape"].median():.1f}%')

    step(f'SHAP for top {TOP_N} items')
    top_items  = eval_df.nlargest(TOP_N, 'mape')['unique_id'].tolist()
    actual_map = dict(zip(eval_df['unique_id'], eval_df['actual']))
    explainer  = make_explainer(model)
    shap_rows  = shap_payloads(explainer, model, week_df, forecast_week, top_items, actual_map)
    check('SHAP rows', len(shap_rows) == TOP_N, f'{len(shap_rows)} rows')
    check('SHAP JSON valid', all(json.loads(r['payload']) for r in shap_rows))

    step(f'Counterfactual for top {TOP_N} items')
    cf_rows = counterfactual_payloads(model, week_df, forecast_week, top_items, actual_map)
    check('Counterfactual rows', len(cf_rows) == TOP_N, f'{len(cf_rows)} rows')

    step(f'Contrastive for top {TOP_N} items')
    ct_rows = contrastive_payloads(explainer, week_df, forecast_week, top_items, eval_df, conn)
    check('Contrastive ran', True, f'{len(ct_rows)} rows (0 expected -- no prior eval history)')

    step('SQLite write + read back (keyed on forecast_week)')
    insert_forecasts(conn, [
        {'week_id': forecast_week, 'item_id': r['unique_id'], 'h1': r['h1'], 'trained_at': 'smoke-test'}
        for r in fcst_df.to_dict('records')
    ])
    insert_evaluations(conn, [
        {'week_id': forecast_week, 'item_id': r['unique_id'],
         'h1_mape': r['mape'], 'h1_mae': r['mae'], 'is_bad_week': 1, 'mape_zscore': 2.0}
        for r in eval_df.to_dict('records')
    ])
    insert_xai(conn, shap_rows + cf_rows)

    # Cross-join check: xai_results and evaluations share the same week_id
    xai_weeks = set(r['week_id'] for r in (shap_rows + cf_rows))
    eval_weeks = {forecast_week}
    check('XAI and eval week keys match', xai_weeks == eval_weeks,
          f'xai={xai_weeks}  eval={eval_weeks}')

    summary = week_summary(conn)
    conn.close()
    check('Summary readable', len(summary) == 1,
          f'{int(summary["n_items"].iloc[0])} items | avg MAPE {summary["avg_mape"].iloc[0]:.1f}%')

    print(f'\n{"=" * 50}')
    print(f'All checks passed  (total {time.perf_counter()-total:.1f}s)')
    print('=' * 50)


if __name__ == '__main__':
    main()
