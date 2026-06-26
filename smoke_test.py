"""
Smoke test: runs ingest (if needed) then one full train/forecast/evaluate/xai cycle.

Usage:
    uv run python smoke_test.py

Expected runtime: ~3 min first run (M5 download + ingest), ~15 sec after.
"""

import sys
import json

from xai_forecast.db import (
    init_db, get_conn, get_weeks,
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
DB_PATH = 'db/smoke_test.db'


def check(label: str, condition: bool, detail: str = '') -> None:
    status = 'OK' if condition else 'FAIL'
    print(f'  [{status}] {label}' + (f' — {detail}' if detail else ''))
    if not condition:
        sys.exit(1)


def main() -> None:
    print('=== Smoke test ===\n')

    print('1. Ingest (or reuse) M5 data into SQLite...')
    init_db(DB_PATH)
    conn = get_conn(DB_PATH)
    weeks = get_weeks(conn)

    if not weeks:
        from xai_forecast.features import load_and_prepare
        from xai_forecast.db import insert_features
        df_raw = load_and_prepare('data')
        df_raw['week'] = df_raw['week'].dt.strftime('%Y-%m-%d')
        keep = ['week', 'unique_id', 'y'] + FEATURE_COLS
        insert_features(conn, df_raw[keep])
        weeks = get_weeks(conn)

    check('Weeks in DB', len(weeks) > TRAIN_WINDOW, f'{len(weeks)} weeks')

    print('\n2. Train on 3-year window...')
    cutoff = weeks[TRAIN_WINDOW]
    window_start = weeks[0]
    train_df = load_features_window(conn, window_start, cutoff)
    model = train_model(train_df)
    check('Model trained', model is not None, f'{len(train_df):,} rows')
    check('Feature importances', len(model.feature_importances_) == len(FEATURE_COLS))

    print('\n3. Forecast next week (h=1)...')
    forecast_week = weeks[TRAIN_WINDOW + 1]
    week_df = load_features_week(conn, forecast_week)
    fcst_df = make_forecasts(model, week_df, forecast_week)
    check('Forecasts returned', len(fcst_df) > 0, f'{len(fcst_df)} rows')
    check('No negatives', (fcst_df['h1'] >= 0).all())
    print(f'    Avg predicted: {fcst_df["h1"].mean():.1f}  |  Median: {fcst_df["h1"].median():.1f}')

    print('\n4. Evaluate...')
    actuals = week_df[['unique_id', 'y']]
    eval_df = evaluate_h1(fcst_df, actuals)
    eval_df['cutoff_week'] = cutoff
    check('Eval rows', len(eval_df) > 0, f'{len(eval_df)} rows')
    check('MAPE in range', eval_df['mape'].between(0, 1000).all())
    print(f'    Avg MAPE: {eval_df["mape"].mean():.1f}%  |  Median: {eval_df["mape"].median():.1f}%')

    print('\n5. XAI on top 5 worst items...')
    top_items = eval_df.nlargest(TOP_N, 'mape')['unique_id'].tolist()
    actual_map = dict(zip(eval_df['unique_id'], eval_df['actual']))
    explainer = make_explainer(model)

    shap_rows = shap_payloads(explainer, model, week_df, forecast_week, top_items, actual_map)
    check('SHAP rows', len(shap_rows) == TOP_N, f'{len(shap_rows)} rows')
    check('SHAP JSON valid', all(json.loads(r['payload']) for r in shap_rows))

    cf_rows = counterfactual_payloads(model, week_df, forecast_week, top_items, actual_map)
    check('Counterfactual rows', len(cf_rows) == TOP_N, f'{len(cf_rows)} rows')

    ct_rows = contrastive_payloads(explainer, week_df, forecast_week, top_items, eval_df, conn)
    check('Contrastive ran', True, f'{len(ct_rows)} rows (0 expected — no prior history yet)')

    print('\n6. SQLite write/read...')
    insert_forecasts(conn, [
        {'week_id': cutoff, 'item_id': r['unique_id'], 'h1': r['h1'], 'trained_at': '2024-01-01'}
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
    check('Summary readable', len(summary) == 1)
    print(f'    {int(summary["n_items"].iloc[0])} items, avg MAPE {summary["avg_mape"].iloc[0]:.1f}%')

    print('\n=== All checks passed ===')


if __name__ == '__main__':
    main()
