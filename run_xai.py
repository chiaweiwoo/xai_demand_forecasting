"""
XAI computation: SHAP, counterfactual, contrastive.

Reads from: db/forecasting.db (evaluations, features)
            db/week_to_cutoff.json
            models/checkpoint_*.lgbm
Writes to:  db/forecasting.db (xai_results)

Run backtest.py first to produce the checkpoint models and week_to_cutoff.json.
Safe to re-run — clears xai_results at start.

Coverage: all valid SKUs per bad week (non-pre-launch features, positive actual sales).
Workers: one ProcessPoolExecutor process per bad week. Workers are read-only;
         the main process collects all rows and does one batch DB write at the end.

Next: uv run python generate_insights.py
"""

import json
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from tqdm import tqdm

from xai_forecast.db import get_conn, insert_xai
from xai_forecast.features import FEATURE_COLS
from xai_forecast.xai import make_explainer, shap_payloads, counterfactual_payloads, contrastive_payloads

DB_PATH    = 'db/forecasting.db'
MODELS_DIR = Path('models')


def _process_week(args: tuple) -> tuple[str, list[dict], int]:
    """
    ProcessPoolExecutor worker — must be top-level (not a closure) for Windows spawn pickling.

    Opens its own SQLite connection (read-only usage; WAL-safe concurrent reads).
    Loads its own LightGBM checkpoint from disk (Boosters don't pickle across processes).
    Returns (forecast_week, xai_rows, n_items). No DB writes happen here.
    """
    forecast_week, xai_cutoff, db_path, models_dir_str = args

    # Worker imports are inside the function so they work cleanly under Windows spawn.
    import sqlite3 as _sqlite3
    import lightgbm as _lgb
    import pandas as _pd
    from pathlib import Path as _Path
    from xai_forecast.features import FEATURE_COLS as _FEATURE_COLS
    from xai_forecast.xai import (
        make_explainer as _make_explainer,
        shap_payloads as _shap_payloads,
        counterfactual_payloads as _cf_payloads,
        contrastive_payloads as _cont_payloads,
    )

    # Plain sqlite3 connection — schema already applied by main process (WAL reads are safe).
    conn = _sqlite3.connect(db_path, timeout=30)
    conn.row_factory = _sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')

    # Full eval history needed for contrastive reference-week lookup.
    all_evals_df = _pd.read_sql(
        'SELECT week_id as forecast_week, item_id as unique_id, h1_mape as mape, h1_mae as mae '
        'FROM evaluations ORDER BY week_id',
        conn,
    )

    # All feature rows for this week.
    week_df = _pd.read_sql(
        'SELECT * FROM features WHERE week = ?',
        conn, params=(forecast_week,),
    )

    # Valid items: must have an evaluation row (real forecast, positive actual via evaluate_h1)
    # AND at least one non-NaN feature column (excludes pre-launch rows imputed to zero).
    week_evals    = all_evals_df[all_evals_df['forecast_week'] == forecast_week]
    evaluated_uids = set(week_evals['unique_id'])
    valid_mask = (
        week_df['unique_id'].isin(evaluated_uids) &
        week_df[_FEATURE_COLS].notna().any(axis=1)
    )
    valid_items = week_df[valid_mask]['unique_id'].tolist()

    if not valid_items:
        conn.close()
        return forecast_week, [], 0

    # Load model checkpoint from disk — each worker loads only what it needs.
    model_path = _Path(models_dir_str) / f'checkpoint_{xai_cutoff}.lgbm'
    model      = _lgb.Booster(model_file=str(model_path))
    explainer  = _make_explainer(model)

    actual_map = dict(zip(week_df['unique_id'], week_df['y']))

    shap_rows, shap_cache = _shap_payloads(
        explainer, model, week_df, forecast_week, valid_items, actual_map,
    )
    cf_rows   = _cf_payloads(model, week_df, forecast_week, valid_items, actual_map)
    cont_rows = _cont_payloads(
        explainer, week_df, forecast_week, valid_items, all_evals_df, conn, shap_cache,
    )

    conn.close()
    return forecast_week, shap_rows + cf_rows + cont_rows, len(valid_items)


def main() -> None:
    cutoff_path = Path('db/week_to_cutoff.json')
    if not cutoff_path.exists():
        raise FileNotFoundError(f'{cutoff_path} not found. Run backtest.py first.')

    week_to_cutoff: dict[str, str] = json.loads(cutoff_path.read_text())

    if not list(MODELS_DIR.glob('checkpoint_*.lgbm')):
        raise FileNotFoundError(
            f'No checkpoint models found in {MODELS_DIR}/. Run backtest.py first.'
        )

    conn = get_conn(DB_PATH)

    bad_weeks = sorted(
        pd.read_sql(
            'SELECT DISTINCT week_id FROM evaluations WHERE is_bad_week=1', conn
        )['week_id'].tolist()
    )
    if not bad_weeks:
        print('No bad weeks found. Run backtest.py first.')
        conn.close()
        return

    skipped = [w for w in bad_weeks if w not in week_to_cutoff]
    if skipped:
        print(f'  Warning: {len(skipped)} weeks have no cutoff mapping — skipped: {skipped}')

    work_items = [
        (week, week_to_cutoff[week], DB_PATH, str(MODELS_DIR))
        for week in bad_weeks
        if week in week_to_cutoff
    ]

    print(f'{len(bad_weeks)} bad weeks — full SKU coverage (all valid non-pre-launch items)')

    # Clean slate for XAI results.
    conn.execute('DELETE FROM xai_results')
    conn.commit()

    n_workers = min(os.cpu_count() or 4, len(work_items))
    print(f'Using {n_workers} worker processes\n')

    all_rows: list[dict] = []
    total_items = 0
    week_counts: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_week, args): args[0] for args in work_items}
        with tqdm(total=len(futures), desc='XAI weeks') as pbar:
            for future in as_completed(futures):
                week = futures[future]
                try:
                    fw, rows, n_items = future.result()
                    all_rows.extend(rows)
                    total_items += n_items
                    week_counts[fw] = n_items
                    pbar.set_postfix({'week': fw, 'items': n_items})
                except Exception as exc:
                    print(f'\n  ERROR processing {week}: {exc}')
                pbar.update(1)

    if all_rows:
        # Chunk the insert to avoid potential memory pressure on very large row sets.
        CHUNK = 10_000
        for i in range(0, len(all_rows), CHUNK):
            insert_xai(conn, all_rows[i:i + CHUNK])

    conn.close()

    n_shap = sum(1 for r in all_rows if r['xai_type'] == 'shap')
    n_cf   = sum(1 for r in all_rows if r['xai_type'] == 'counterfactual')
    n_cont = sum(1 for r in all_rows if r['xai_type'] == 'contrastive')
    avg_items = total_items // max(len(work_items), 1)

    print(f'\nDone -> {DB_PATH}')
    print(f'  {len(work_items)} bad weeks | avg {avg_items} valid items/week')
    print(f'  SHAP: {n_shap:,} | Counterfactual: {n_cf:,} | Contrastive: {n_cont:,} | Total: {len(all_rows):,}')
    print('Next: uv run python generate_insights.py')


if __name__ == '__main__':
    main()
