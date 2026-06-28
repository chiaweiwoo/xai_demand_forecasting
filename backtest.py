"""
Sliding-window backtest: forecast + evaluate only.

Run ingest.py and build_features.py first.

Outputs:
  db/forecasting.db       -> forecasts, evaluations tables
  models/checkpoint_*.lgbm -> one LightGBM checkpoint per retrain cutoff
  db/week_to_cutoff.json  -> maps forecast_week -> retrain cutoff (needed by run_xai.py)

Next steps:
  uv run python run_xai.py           # SHAP / counterfactual / contrastive
  uv run python generate_insights.py    # evidence-first insights
  uv run python data_quality.py
  uv run streamlit run app.py
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from xai_forecast.db import (
    get_conn, get_all_weeks,
    load_features_window, load_features_week,
    insert_forecasts, insert_evaluations,
)
from xai_forecast.features import FEATURE_COLS
from xai_forecast.train import train_model
from xai_forecast.forecast import make_forecasts
from xai_forecast.evaluate import evaluate_h1, flag_bad_weeks

TRAIN_WINDOW = 156   # 3 years
RETRAIN_FREQ = 4
DB_PATH      = 'db/forecasting.db'
MODELS_DIR   = Path('models')


def main() -> None:
    conn = get_conn(DB_PATH)
    weeks = get_all_weeks(conn)

    if not weeks:
        print('No data. Run: uv run python ingest.py && uv run python build_features.py')
        return

    n_features = conn.execute('SELECT COUNT(*) FROM features').fetchone()[0]
    if n_features == 0:
        print('Feature store empty. Run: uv run python build_features.py')
        conn.close()
        return

    print(f'{len(weeks)} weeks ({weeks[0]} -> {weeks[-1]})')
    # Exclude weeks[-2:] — M5 eval file ends mid-week; last two weeks produce spurious spikes.
    backtest_weeks = weeks[TRAIN_WINDOW:-2]
    print(f'Backtest: {len(backtest_weeks)} weeks | ~{len(backtest_weeks) // RETRAIN_FREQ} retrains\n')

    # Clean slate for forecast/evaluation tables only.
    # xai_results and insights are managed by run_xai.py / generate_insights.py.
    conn.executescript('DELETE FROM forecasts; DELETE FROM evaluations;')
    conn.commit()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model: lgb.LGBMRegressor | None = None
    all_evals: list[pd.DataFrame] = []
    week_to_cutoff: dict[str, str] = {}
    last_retrain_cutoff: str | None = None
    n_nan_imputed = 0

    for i, cutoff in enumerate(tqdm(backtest_weeks, desc='Backtesting')):
        step = TRAIN_WINDOW + i
        forecast_week = weeks[step + 1]

        if i % RETRAIN_FREQ == 0:
            last_retrain_cutoff = cutoff
            window_start = weeks[step - TRAIN_WINDOW]
            train_df     = load_features_window(conn, window_start, cutoff).dropna(subset=FEATURE_COLS)
            model        = train_model(train_df)
            model.booster_.save_model(str(MODELS_DIR / f'checkpoint_{last_retrain_cutoff}.lgbm'))

        week_to_cutoff[forecast_week] = last_retrain_cutoff

        week_df = load_features_week(conn, forecast_week)
        n_nan_imputed += int((week_df[FEATURE_COLS].isnull().all(axis=1)).sum())

        preds = make_forecasts(model, week_df, forecast_week)
        insert_forecasts(conn, [
            {'week_id': forecast_week, 'item_id': r['unique_id'],
             'h1': r['h1'], 'trained_at': datetime.utcnow().isoformat()}
            for r in preds.to_dict('records')
        ])

        actuals = week_df[['unique_id', 'y']]
        eval_df = evaluate_h1(preds, actuals)
        eval_df['forecast_week'] = forecast_week
        all_evals.append(eval_df)

    if n_nan_imputed > 0:
        print(f'\n  Note: {n_nan_imputed:,} pre-launch SKU-week rows had all-NaN features -> imputed to 0.')

    print('\nFlagging bad weeks (WMAPE z-score, prior-weeks-only baseline)...')
    all_evals_df = pd.concat(all_evals, ignore_index=True)
    week_flags   = flag_bad_weeks(all_evals_df)
    bad_weeks    = week_flags[week_flags['is_bad_week']]['forecast_week'].tolist()

    zscore_map = week_flags.set_index('forecast_week')['zscore'].to_dict()
    is_bad_map = week_flags.set_index('forecast_week')['is_bad_week'].to_dict()

    insert_evaluations(conn, [
        {'week_id': r['forecast_week'], 'item_id': r['unique_id'],
         'h1_mape': r['mape'], 'h1_mae': r['mae'],
         'is_bad_week': int(is_bad_map.get(r['forecast_week'], False)),
         'mape_zscore': float(z) if pd.notna(z := zscore_map.get(r['forecast_week'], 0)) else 0.0}
        for r in all_evals_df.to_dict('records')
    ])
    print(f'  {len(bad_weeks)} bad weeks out of {len(backtest_weeks)}')

    # Save week->cutoff mapping so run_xai.py knows which checkpoint to use per forecast week.
    cutoff_path = Path('db/week_to_cutoff.json')
    cutoff_path.parent.mkdir(parents=True, exist_ok=True)
    cutoff_path.write_text(json.dumps(week_to_cutoff))
    print(f'  Saved {len(week_to_cutoff)} week->cutoff mappings -> {cutoff_path}')
    print(f'  Saved {len(list(MODELS_DIR.glob("*.lgbm")))} checkpoint models -> {MODELS_DIR}/')

    conn.close()
    print(f'\nDone -> {DB_PATH}')
    print('Next: uv run python run_xai.py')


if __name__ == '__main__':
    main()
