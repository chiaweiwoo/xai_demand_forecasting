"""
XAI computation: SHAP, counterfactual, contrastive.

Reads from: db/forecasting.db (evaluations, features)
            db/week_to_cutoff.json
            models/checkpoint_*.lgbm
Writes to:  db/forecasting.db (xai_results)

Run backtest.py first to produce the checkpoint models and week_to_cutoff.json.
Safe to re-run — clears xai_results at start.

Next: uv run python generate_insights.py
"""

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from tqdm import tqdm

from xai_forecast.db import get_conn, load_features_week, insert_xai
from xai_forecast.xai import make_explainer, shap_payloads, counterfactual_payloads, contrastive_payloads

DB_PATH    = 'db/forecasting.db'
MODELS_DIR = Path('models')
TOP_N_XAI  = 50


def _load_checkpoints() -> dict[str, lgb.Booster]:
    files = sorted(MODELS_DIR.glob('checkpoint_*.lgbm'))
    if not files:
        raise FileNotFoundError(
            f'No checkpoint models found in {MODELS_DIR}/. Run backtest.py first.'
        )
    return {f.stem.replace('checkpoint_', ''): lgb.Booster(model_file=str(f)) for f in files}


def main() -> None:
    cutoff_path = Path('db/week_to_cutoff.json')
    if not cutoff_path.exists():
        raise FileNotFoundError(f'{cutoff_path} not found. Run backtest.py first.')

    week_to_cutoff: dict[str, str] = json.loads(cutoff_path.read_text())

    print(f'Loading checkpoint models from {MODELS_DIR}/ ...')
    all_models = _load_checkpoints()
    print(f'  {len(all_models)} checkpoints loaded')

    conn = get_conn(DB_PATH)

    # Read bad weeks and full eval history from DB
    evals_df = pd.read_sql(
        'SELECT week_id as forecast_week, item_id as unique_id, h1_mape as mape, h1_mae as mae '
        'FROM evaluations ORDER BY week_id',
        conn,
    )
    bad_weeks = (
        evals_df[evals_df['forecast_week'].isin(
            pd.read_sql(
                'SELECT DISTINCT week_id FROM evaluations WHERE is_bad_week=1', conn
            )['week_id'].tolist()
        )]['forecast_week'].unique().tolist()
    )
    bad_weeks = sorted(bad_weeks)
    print(f'\n{len(bad_weeks)} bad weeks to process')

    # Clean slate for XAI results (insights managed separately by generate_insights.py)
    conn.execute('DELETE FROM xai_results')
    conn.commit()

    explainers_cache: dict[str, object] = {}

    for forecast_week in tqdm(bad_weeks, desc='XAI'):
        xai_cutoff = week_to_cutoff.get(forecast_week)
        if xai_cutoff is None or xai_cutoff not in all_models:
            print(f'  Warning: no checkpoint for {forecast_week} (cutoff={xai_cutoff}) — skipped')
            continue

        xai_model = all_models[xai_cutoff]
        if xai_cutoff not in explainers_cache:
            explainers_cache[xai_cutoff] = make_explainer(xai_model)
        xai_explainer = explainers_cache[xai_cutoff]

        week_df    = load_features_week(conn, forecast_week)
        week_evals = evals_df[evals_df['forecast_week'] == forecast_week]
        top_items  = week_evals.nlargest(TOP_N_XAI, 'mape')['unique_id'].tolist()

        # actual_map from feature store (y column = weekly sales)
        actual_map = dict(zip(week_df['unique_id'], week_df['y']))

        shap_rows, shap_cache = shap_payloads(
            xai_explainer, xai_model, week_df, forecast_week, top_items, actual_map
        )
        cf_rows   = counterfactual_payloads(xai_model, week_df, forecast_week, top_items, actual_map)
        cont_rows = contrastive_payloads(
            xai_explainer, week_df, forecast_week, top_items, evals_df, conn, shap_cache
        )

        xai_rows = shap_rows + cf_rows + cont_rows
        if xai_rows:
            insert_xai(conn, xai_rows)

    conn.close()
    print(f'\nDone -> {DB_PATH}')
    print('Next: uv run python generate_insights.py')


if __name__ == '__main__':
    main()
