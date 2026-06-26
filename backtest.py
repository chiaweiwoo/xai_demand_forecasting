"""
Full backtesting simulation. Run ingest.py first.

Usage:
    uv run python backtest.py

Training window: fixed 3-year (156-week) sliding window.
Retrain every 4 weeks. Forecast horizon = 1.
All data read from SQLite — no datasetsforecast import at runtime.
"""

from datetime import datetime
from tqdm import tqdm
import pandas as pd

from xai_forecast.db import (
    init_db, get_conn,
    load_features_window, load_features_week, get_weeks,
    insert_forecasts, insert_evaluations, insert_xai,
)
from xai_forecast.train import train_model
from xai_forecast.forecast import make_forecasts
from xai_forecast.evaluate import evaluate_h1, flag_bad_weeks
from xai_forecast.xai import make_explainer, shap_payloads, counterfactual_payloads, contrastive_payloads

TRAIN_WINDOW = 156   # 3 years
RETRAIN_FREQ = 4
TOP_N_XAI = 50
DB_PATH = 'db/forecasting.db'


def main() -> None:
    conn = get_conn(DB_PATH)
    weeks = get_weeks(conn)

    if not weeks:
        print('No data found. Run: uv run python ingest.py')
        return

    print(f'Loaded {len(weeks)} weeks from SQLite ({weeks[0]} → {weeks[-1]})')

    backtest_weeks = weeks[TRAIN_WINDOW:-1]
    print(f'Backtest: {len(backtest_weeks)} weeks | ~{len(backtest_weeks) // RETRAIN_FREQ} retrains\n')

    model = None
    all_evals: list[pd.DataFrame] = []

    for i, cutoff in enumerate(tqdm(backtest_weeks, desc='Backtesting')):
        step = TRAIN_WINDOW + i

        if i % RETRAIN_FREQ == 0:
            window_start = weeks[step - TRAIN_WINDOW]
            train_df = load_features_window(conn, window_start, cutoff)
            model = train_model(train_df)

        forecast_week = weeks[step + 1]
        fcst_df = load_features_week(conn, forecast_week)
        preds = make_forecasts(model, fcst_df, forecast_week)

        trained_at = datetime.utcnow().isoformat()
        insert_forecasts(conn, [
            {'week_id': cutoff, 'item_id': r['unique_id'], 'h1': r['h1'], 'trained_at': trained_at}
            for r in preds.to_dict('records')
        ])

        actuals = fcst_df[['unique_id', 'y']].rename(columns={'y': 'y'})
        eval_df = evaluate_h1(preds, actuals)
        eval_df['cutoff_week'] = cutoff
        all_evals.append(eval_df)

    print('\nFlagging bad weeks...')
    all_evals_df = pd.concat(all_evals, ignore_index=True)
    week_flags = flag_bad_weeks(all_evals_df)
    bad_weeks = week_flags[week_flags['is_bad_week']]['cutoff_week'].tolist()

    zscore_map = week_flags.set_index('cutoff_week')['zscore'].to_dict()
    is_bad_map = week_flags.set_index('cutoff_week')['is_bad_week'].to_dict()

    insert_evaluations(conn, [
        {'week_id': r['cutoff_week'], 'item_id': r['unique_id'],
         'h1_mape': r['mape'], 'h1_mae': r['mae'],
         'is_bad_week': int(is_bad_map.get(r['cutoff_week'], False)),
         'mape_zscore': float(zscore_map.get(r['cutoff_week'], 0) or 0)}
        for r in all_evals_df.to_dict('records')
    ])
    print(f'  {len(bad_weeks)} bad weeks out of {len(backtest_weeks)}')

    print('\nComputing XAI...')
    # Retrain on most recent 3-year window for XAI explanations
    train_df = load_features_window(conn, weeks[-TRAIN_WINDOW - 1], weeks[-1])
    model = train_model(train_df)
    explainer = make_explainer(model)

    for cutoff in tqdm(bad_weeks, desc='XAI'):
        step = weeks.index(cutoff)
        forecast_week = weeks[step + 1]
        fcst_df = load_features_week(conn, forecast_week)

        week_evals = all_evals_df[all_evals_df['cutoff_week'] == cutoff]
        top_items = week_evals.nlargest(TOP_N_XAI, 'mape')['unique_id'].tolist()
        actual_map = dict(zip(week_evals['unique_id'], week_evals['actual']))

        xai_rows = (
            shap_payloads(explainer, model, fcst_df, forecast_week, top_items, actual_map)
            + counterfactual_payloads(model, fcst_df, forecast_week, top_items, actual_map)
            + contrastive_payloads(explainer, fcst_df, forecast_week, top_items, all_evals_df, conn)
        )
        if xai_rows:
            insert_xai(conn, xai_rows)

    conn.close()
    print(f'\nDone → {DB_PATH}')
    print('Launch: uv run streamlit run app.py')


if __name__ == '__main__':
    main()
