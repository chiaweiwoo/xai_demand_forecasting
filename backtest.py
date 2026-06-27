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

import json
from collections import defaultdict
from datetime import datetime

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
    insert_forecasts, insert_evaluations, insert_xai, insert_narrative,
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
    conn.executescript(
        'DELETE FROM forecasts; DELETE FROM evaluations; DELETE FROM xai_results; DELETE FROM narratives;'
    )
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
    # Accumulate XAI rows in memory for the narrative phase below.
    xai_data_per_week: dict[str, dict] = {}

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
        cf_rows   = counterfactual_payloads(xai_model, week_df, forecast_week, top_items, actual_map)
        cont_rows = contrastive_payloads(xai_explainer, week_df, forecast_week, top_items, all_evals_df, conn, shap_cache)
        xai_rows  = shap_rows + cf_rows + cont_rows
        if xai_rows:
            insert_xai(conn, xai_rows)

        xai_data_per_week[forecast_week] = {
            'shap_rows': shap_rows,
            'cf_rows': cf_rows,
            'cont_rows': cont_rows,
            'top_items': top_items[:5],
            'week_evals': week_evals,
        }

    # ── Narrative generation (requires DEEPSEEK_API_KEY in env) ──────────────
    try:
        from xai_forecast.narrate import (
            DeepSeekNarrator,
            WEEK_NARRATIVE_PROMPT, ITEM_NARRATIVE_PROMPT, EXECUTIVE_NARRATIVE_PROMPT,
            build_week_dossier, build_item_dossier, build_executive_dossier,
        )
        narrator = DeepSeekNarrator()
    except ImportError:
        narrator = None  # type: ignore[assignment]

    if narrator and narrator.available:
        print(f'\nGenerating LLM narratives ({len(bad_weeks)} bad weeks + 1 executive)...')

        for forecast_week in tqdm(bad_weeks, desc='Week narratives'):
            data = xai_data_per_week.get(forecast_week, {})
            shap_rows = data.get('shap_rows', [])
            if not shap_rows:
                continue

            week_evals = data.get('week_evals', pd.DataFrame())
            zscore = zscore_map.get(forecast_week)

            week_doss = build_week_dossier(forecast_week, shap_rows, zscore, len(week_evals))
            week_narr = narrator.generate(WEEK_NARRATIVE_PROMPT, week_doss)
            if week_narr:
                insert_narrative(conn, 'week', forecast_week, week_narr, narrator.model_id)

            # Item-level narratives for the 5 worst SKUs per bad week
            shap_by_item = {r['item_id']: r for r in shap_rows}
            cf_by_item   = {r['item_id']: r for r in data.get('cf_rows', [])}
            cont_by_item = {r['item_id']: r for r in data.get('cont_rows', [])}

            for item_id in data.get('top_items', []):
                shap_p = json.loads(shap_by_item[item_id]['payload']) if item_id in shap_by_item else None
                cf_p   = json.loads(cf_by_item[item_id]['payload'])   if item_id in cf_by_item   else None
                cont_p = json.loads(cont_by_item[item_id]['payload']) if item_id in cont_by_item else None
                item_doss = build_item_dossier(forecast_week, item_id, shap_p, cf_p, cont_p)
                item_narr = narrator.generate(ITEM_NARRATIVE_PROMPT, item_doss)
                if item_narr:
                    insert_narrative(conn, 'item', f'{forecast_week}::{item_id}', item_narr, narrator.model_id)

        # Executive synthesis — aggregate recurring drivers across all bad weeks
        feature_counts: dict[str, int] = defaultdict(int)
        total_payloads = 0
        for week_data in xai_data_per_week.values():
            for row in week_data.get('shap_rows', []):
                total_payloads += 1
                p = json.loads(row['payload'])
                for f in p.get('top_features', []):
                    feature_counts[f['feature']] += 1

        drivers_list = sorted(
            [
                {'feature': feat, 'count': cnt,
                 'pct_payloads': round(cnt / total_payloads * 100, 1) if total_payloads else 0}
                for feat, cnt in feature_counts.items()
            ],
            key=lambda x: x['count'], reverse=True,
        )
        exec_doss = build_executive_dossier(drivers_list, len(bad_weeks), len(backtest_weeks))
        exec_narr = narrator.generate(EXECUTIVE_NARRATIVE_PROMPT, exec_doss)
        if exec_narr:
            insert_narrative(conn, 'executive', 'overall', exec_narr, narrator.model_id)

        print('  Narratives saved to DB.')
    else:
        print('\nNo DEEPSEEK_API_KEY — narrative generation skipped.')
        print('  Set DEEPSEEK_API_KEY in .env and re-run to generate narratives.')

    conn.close()
    print(f'\nDone -> {DB_PATH}')
    print('Next: uv run python data_quality.py')
    print('Then: uv run streamlit run app.py')


if __name__ == '__main__':
    main()
