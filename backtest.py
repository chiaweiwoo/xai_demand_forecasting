"""
Full backtesting simulation. Run ingest.py first.

Usage:
    uv run python backtest.py

At each iteration: query raw tables -> compute features in memory -> train -> forecast.
No features table -- feature engineering is part of the pipeline, not precomputed.

Week key convention: all tables (forecasts, evaluations, xai_results) are keyed on
forecast_week -- the week the failure was observed, not the training cutoff. This is
the natural "week X" a leader would point at.

XAI model note: SHAP and counterfactuals are computed with a model retrained on the
most recent 3-year window, not the model that produced each week's forecast. For a
portfolio of 125 weeks this is a necessary tradeoff -- the explanations reflect the
model's learned feature relationships, which are stable, but strictly they come from
a different model instance than the one that originally erred.
"""

from datetime import datetime
from tqdm import tqdm
import pandas as pd

from xai_forecast.db import (
    get_conn, get_all_weeks, load_raw_window,
    insert_forecasts, insert_evaluations, insert_xai,
)
from xai_forecast.features import compute_features, FEATURE_COLS, HISTORY_BUFFER
from xai_forecast.train import train_model
from xai_forecast.forecast import make_forecasts
from xai_forecast.evaluate import evaluate_h1, flag_bad_weeks
from xai_forecast.xai import make_explainer, shap_payloads, counterfactual_payloads, contrastive_payloads

TRAIN_WINDOW = 156   # 3 years
RETRAIN_FREQ = 4
TOP_N_XAI    = 50
DB_PATH      = 'db/forecasting.db'


def main() -> None:
    conn = get_conn(DB_PATH)
    weeks = get_all_weeks(conn)

    if not weeks:
        print('No data. Run: uv run python ingest.py')
        return

    print(f'{len(weeks)} weeks ({weeks[0]} -> {weeks[-1]})')
    backtest_weeks = weeks[TRAIN_WINDOW:-1]
    print(f'Backtest: {len(backtest_weeks)} weeks | ~{len(backtest_weeks) // RETRAIN_FREQ} retrains\n')

    model = None
    all_evals: list[pd.DataFrame] = []

    for i, cutoff in enumerate(tqdm(backtest_weeks, desc='Backtesting')):
        step = TRAIN_WINDOW + i
        forecast_week = weeks[step + 1]

        if i % RETRAIN_FREQ == 0:
            window_start = weeks[step - TRAIN_WINDOW]
            buffer_start = weeks[max(0, step - TRAIN_WINDOW - HISTORY_BUFFER)]
            raw_df       = load_raw_window(conn, buffer_start, cutoff)
            features_df  = compute_features(raw_df)
            train_df     = features_df[features_df['week'] > window_start].dropna(subset=FEATURE_COLS)
            model        = train_model(train_df)

        buf_start = weeks[max(0, step + 1 - HISTORY_BUFFER)]
        raw_fcst  = load_raw_window(conn, buf_start, forecast_week)
        feat_fcst = compute_features(raw_fcst)
        week_df   = feat_fcst[feat_fcst['week'] == forecast_week]
        preds     = make_forecasts(model, week_df, forecast_week)

        # Store under forecast_week -- the week the failure was observed
        insert_forecasts(conn, [
            {'week_id': forecast_week, 'item_id': r['unique_id'],
             'h1': r['h1'], 'trained_at': datetime.utcnow().isoformat()}
            for r in preds.to_dict('records')
        ])

        actuals = week_df[['unique_id', 'y']]
        eval_df = evaluate_h1(preds, actuals)
        eval_df['forecast_week'] = forecast_week
        all_evals.append(eval_df)

    print('\nFlagging bad weeks (WMAPE-based)...')
    all_evals_df = pd.concat(all_evals, ignore_index=True)
    week_flags   = flag_bad_weeks(all_evals_df)
    bad_weeks    = week_flags[week_flags['is_bad_week']]['forecast_week'].tolist()

    zscore_map = week_flags.set_index('forecast_week')['zscore'].to_dict()
    is_bad_map = week_flags.set_index('forecast_week')['is_bad_week'].to_dict()

    insert_evaluations(conn, [
        {'week_id': r['forecast_week'], 'item_id': r['unique_id'],
         'h1_mape': r['mape'], 'h1_mae': r['mae'],
         'is_bad_week': int(is_bad_map.get(r['forecast_week'], False)),
         'mape_zscore': float(zscore_map.get(r['forecast_week'], 0) or 0)}
        for r in all_evals_df.to_dict('records')
    ])
    print(f'  {len(bad_weeks)} bad weeks out of {len(backtest_weeks)}')

    print('\nComputing XAI...')
    # Single model on most recent 3-year window (see module docstring for caveat)
    buf_start = weeks[max(0, len(weeks) - TRAIN_WINDOW - HISTORY_BUFFER - 1)]
    raw_df    = load_raw_window(conn, buf_start, weeks[-1])
    feat_df   = compute_features(raw_df)
    train_df  = feat_df[feat_df['week'] > weeks[-TRAIN_WINDOW - 1]].dropna(subset=FEATURE_COLS)
    model     = train_model(train_df)
    explainer = make_explainer(model)

    # bad_weeks contains forecast_weeks — iterate directly
    for forecast_week in tqdm(bad_weeks, desc='XAI'):
        step      = weeks.index(forecast_week)
        buf_start = weeks[max(0, step - HISTORY_BUFFER)]
        raw_fcst  = load_raw_window(conn, buf_start, forecast_week)
        feat_fcst = compute_features(raw_fcst)
        week_df   = feat_fcst[feat_fcst['week'] == forecast_week]

        week_evals = all_evals_df[all_evals_df['forecast_week'] == forecast_week]
        top_items  = week_evals.nlargest(TOP_N_XAI, 'mape')['unique_id'].tolist()
        actual_map = dict(zip(week_evals['unique_id'], week_evals['actual']))

        xai_rows = (
            shap_payloads(explainer, model, week_df, forecast_week, top_items, actual_map)
            + counterfactual_payloads(model, week_df, forecast_week, top_items, actual_map)
            + contrastive_payloads(explainer, week_df, forecast_week, top_items, week_evals, conn)
        )
        if xai_rows:
            insert_xai(conn, xai_rows)

    conn.close()
    print(f'\nDone -> {DB_PATH}')
    print('Launch: uv run streamlit run app.py')


if __name__ == '__main__':
    main()
