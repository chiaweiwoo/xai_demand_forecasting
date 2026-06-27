"""
Full backtesting simulation. Run ingest.py and build_features.py first.

Usage:
    uv run python backtest.py

At each iteration: load precomputed features from the feature store -> train -> forecast.
Each retrain checkpoint is kept in memory so XAI explanations come from the same model
that produced each week's forecast (at 4-week retrain granularity).

Week key convention: all tables (forecasts, evaluations, xai_results) are keyed on
forecast_week -- the week the failure was observed, not the training cutoff. This is
the natural "week X" a leader would point at.
"""

from datetime import datetime
from tqdm import tqdm
import pandas as pd
import lightgbm as lgb

from xai_forecast.db import (
    get_conn, get_all_weeks,
    load_features_window, load_features_week,
    insert_forecasts, insert_evaluations, insert_xai,
)
from xai_forecast.features import FEATURE_COLS
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
        print('No data. Run: uv run python ingest.py && uv run python build_features.py')
        return

    n_features = conn.execute('SELECT COUNT(*) FROM features').fetchone()[0]
    if n_features == 0:
        print('Feature store empty. Run: uv run python build_features.py')
        conn.close()
        return

    print(f'{len(weeks)} weeks ({weeks[0]} -> {weeks[-1]})')
    # Exclude weeks[-1] — the M5 eval file ends mid-week (only 2 days), making it a partial
    # week with artificially low sales. Forecasting it produces spurious bad-week flags.
    backtest_weeks = weeks[TRAIN_WINDOW:-2]
    print(f'Backtest: {len(backtest_weeks)} weeks | ~{len(backtest_weeks) // RETRAIN_FREQ} retrains\n')

    # Clean slate — ensures no orphan rows from previous or partial runs
    conn.executescript('DELETE FROM forecasts; DELETE FROM evaluations; DELETE FROM xai_results;')
    conn.commit()

    model: lgb.LGBMRegressor | None = None
    all_evals: list[pd.DataFrame] = []
    # Per-checkpoint model store: retrain_cutoff → model
    # Allows XAI to use the exact model that produced each week's forecast.
    all_models: dict[str, lgb.LGBMRegressor] = {}
    week_to_cutoff: dict[str, str] = {}  # forecast_week → retrain cutoff used
    last_retrain_cutoff: str | None = None
    n_nan_imputed = 0  # pre-launch SKUs forecasted from all-zero features

    for i, cutoff in enumerate(tqdm(backtest_weeks, desc='Backtesting')):
        step = TRAIN_WINDOW + i
        forecast_week = weeks[step + 1]

        if i % RETRAIN_FREQ == 0:
            last_retrain_cutoff = cutoff
            window_start = weeks[step - TRAIN_WINDOW]
            train_df     = load_features_window(conn, window_start, cutoff).dropna(subset=FEATURE_COLS)
            model        = train_model(train_df)
            all_models[last_retrain_cutoff] = model

        week_to_cutoff[forecast_week] = last_retrain_cutoff
        week_df = load_features_week(conn, forecast_week)

        # Count pre-launch SKUs (all lag features NaN → imputed to 0 by make_forecasts)
        n_nan_rows = int((week_df[FEATURE_COLS].isnull().all(axis=1)).sum())
        n_nan_imputed += n_nan_rows

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

    if n_nan_imputed > 0:
        print(f'\n  Note: {n_nan_imputed:,} pre-launch SKU-week rows had all-NaN features → '
              f'imputed to 0 by make_forecasts. These are scored against actual if y>0.')

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

    print('\nComputing XAI (per-retrain-checkpoint model)...')
    # Use the model that actually produced each week's forecast (same 4-week retrain granularity).
    # Explainers are cached by cutoff to avoid rebuilding per bad week.
    explainers_cache: dict[str, object] = {}

    for forecast_week in tqdm(bad_weeks, desc='XAI'):
        xai_cutoff = week_to_cutoff.get(forecast_week)
        if xai_cutoff is None or xai_cutoff not in all_models:
            continue

        xai_model = all_models[xai_cutoff]
        if xai_cutoff not in explainers_cache:
            explainers_cache[xai_cutoff] = make_explainer(xai_model)
        xai_explainer = explainers_cache[xai_cutoff]

        week_df    = load_features_week(conn, forecast_week)
        week_evals = all_evals_df[all_evals_df['forecast_week'] == forecast_week]
        top_items  = week_evals.nlargest(TOP_N_XAI, 'mape')['unique_id'].tolist()
        actual_map = dict(zip(week_evals['unique_id'], week_evals['actual']))

        shap_rows, shap_cache = shap_payloads(xai_explainer, xai_model, week_df, forecast_week, top_items, actual_map)
        xai_rows = (
            shap_rows
            + counterfactual_payloads(xai_model, week_df, forecast_week, top_items, actual_map)
            + contrastive_payloads(xai_explainer, week_df, forecast_week, top_items, all_evals_df, conn, shap_cache)
        )
        if xai_rows:
            insert_xai(conn, xai_rows)

    conn.close()
    print(f'\nDone -> {DB_PATH}')
    print('Next: uv run python data_quality.py')
    print('Then: uv run streamlit run app.py')


if __name__ == '__main__':
    main()
