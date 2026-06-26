"""
Full backtesting simulation.

Usage:
    uv run python backtest.py

What it does:
  - Loads M5 data (CA_1 store, ~3k SKUs)
  - Expanding-window backtest: trains LightGBM every 4 weeks, forecasts h=1/2/3
  - Flags bad weeks (avg MAPE > rolling mean + 1.5 SD)
  - Runs SHAP, counterfactual, and contrastive XAI for the top 50 worst items
    in each bad week
  - Writes everything to db/forecasting.db (SQLite)
"""

from datetime import datetime
from tqdm import tqdm
import pandas as pd

from xai_forecast.features import load_and_prepare, FEATURE_COLS
from xai_forecast.db import init_db, get_conn, insert_forecasts, insert_evaluations, insert_xai
from xai_forecast.train import train_model
from xai_forecast.forecast import make_forecasts
from xai_forecast.evaluate import evaluate_h1, flag_bad_weeks
from xai_forecast.xai import make_explainer, shap_payloads, counterfactual_payloads, contrastive_payloads

WARMUP_WEEKS = 52
RETRAIN_FREQ = 4      # retrain every N weeks
TOP_N_XAI = 50        # worst items to explain per bad week
DATA_DIR = 'data'
DB_PATH = 'db/forecasting.db'


def main() -> None:
    print('Loading M5 data...')
    df = load_and_prepare(DATA_DIR)
    weeks = sorted(df['week'].unique())
    n_items = df['unique_id'].nunique()
    print(f'  {n_items} items | {len(weeks)} weeks ({weeks[0].date()} → {weeks[-1].date()})')

    init_db(DB_PATH)
    conn = get_conn(DB_PATH)

    model = None
    all_evals: list[pd.DataFrame] = []

    print(f'\nBacktest (warmup={WARMUP_WEEKS}w, retrain every {RETRAIN_FREQ}w)...')
    backtest_weeks = weeks[WARMUP_WEEKS:-3]

    for i, cutoff in enumerate(tqdm(backtest_weeks)):
        step = i + WARMUP_WEEKS
        forecast_weeks = weeks[step + 1 : step + 4]

        if i % RETRAIN_FREQ == 0:
            train_df = df[df['week'] <= cutoff]
            model = train_model(train_df)

        fcst_df = make_forecasts(model, df, forecast_weeks)

        trained_at = datetime.utcnow().isoformat()
        insert_forecasts(conn, [
            {
                'week_id': str(cutoff.date()),
                'item_id': r['unique_id'],
                'h1': r['h1'], 'h2': r['h2'], 'h3': r['h3'],
                'trained_at': trained_at,
            }
            for r in fcst_df.to_dict('records')
        ])

        h1_actuals = df[df['week'] == forecast_weeks[0]][['unique_id', 'y']].copy()
        eval_df = evaluate_h1(fcst_df, h1_actuals)
        eval_df['cutoff_week'] = cutoff
        all_evals.append(eval_df)

    print('\nFlagging bad weeks...')
    all_evals_df = pd.concat(all_evals, ignore_index=True)
    week_flags = flag_bad_weeks(all_evals_df)
    bad_cutoff_weeks = week_flags[week_flags['is_bad_week']]['cutoff_week'].tolist()

    zscore_map = week_flags.set_index('cutoff_week')['zscore'].to_dict()
    is_bad_map = week_flags.set_index('cutoff_week')['is_bad_week'].to_dict()

    eval_rows = [
        {
            'week_id': str(r['cutoff_week'].date()),
            'item_id': r['unique_id'],
            'h1_mape': r['mape'],
            'h1_mae': r['mae'],
            'is_bad_week': int(is_bad_map.get(r['cutoff_week'], False)),
            'mape_zscore': float(zscore_map.get(r['cutoff_week'], 0) or 0),
        }
        for r in all_evals_df.to_dict('records')
    ]
    insert_evaluations(conn, eval_rows)
    print(f'  {len(bad_cutoff_weeks)} bad weeks flagged out of {len(backtest_weeks)}')

    print('\nComputing XAI (retrain on full data for explanations)...')
    model = train_model(df)
    explainer = make_explainer(model)

    for cutoff in tqdm(bad_cutoff_weeks):
        h1_week = cutoff + pd.Timedelta(weeks=1)
        week_evals = all_evals_df[all_evals_df['cutoff_week'] == cutoff]
        top_items = week_evals.nlargest(TOP_N_XAI, 'mape')['unique_id'].tolist()
        actual_map = dict(zip(week_evals['unique_id'], week_evals['actual']))

        xai_rows = (
            shap_payloads(explainer, model, df, h1_week, top_items, actual_map)
            + counterfactual_payloads(model, df, h1_week, top_items, actual_map)
            + contrastive_payloads(explainer, df, h1_week, top_items, all_evals_df)
        )
        if xai_rows:
            insert_xai(conn, xai_rows)

    conn.close()
    print(f'\nDone → {DB_PATH}')
    print('Launch dashboard: uv run streamlit run app.py')


if __name__ == '__main__':
    main()
