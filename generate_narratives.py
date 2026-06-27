"""
LLM narrative generation.

Reads from: db/forecasting.db (evaluations, xai_results)
Writes to:  db/forecasting.db (narratives)

Requires DEEPSEEK_API_KEY in .env. Run run_xai.py first to populate xai_results.
Safe to re-run — clears narratives at start and regenerates all.

Next: uv run python data_quality.py
"""

import json

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from xai_forecast.db import (
    get_conn, load_all_shap_payloads,
    insert_narrative, load_xai,
)
from xai_forecast.narrate import (
    DeepSeekNarrator,
    WEEK_NARRATIVE_PROMPT, ITEM_NARRATIVE_PROMPT, EXECUTIVE_NARRATIVE_PROMPT,
    build_week_dossier, build_item_dossier, build_executive_dossier,
    compute_recurring_drivers,
)

DB_PATH   = 'db/forecasting.db'
TOP_ITEMS = 5   # item narratives per bad week


def main() -> None:
    conn = get_conn(DB_PATH)

    narrator = DeepSeekNarrator()
    if not narrator.available:
        print('No DEEPSEEK_API_KEY — set it in .env and re-run.')
        conn.close()
        return

    # Read bad weeks and their zscores
    evals_df = pd.read_sql(
        'SELECT DISTINCT week_id, mape_zscore FROM evaluations WHERE is_bad_week=1 ORDER BY week_id',
        conn,
    )
    bad_weeks  = evals_df['week_id'].tolist()
    zscore_map = dict(zip(evals_df['week_id'], evals_df['mape_zscore']))

    if not bad_weeks:
        print('No bad weeks in evaluations table. Run backtest.py + run_xai.py first.')
        conn.close()
        return

    print(f'{len(bad_weeks)} bad weeks to narrate')

    # Full eval history (needed for item narrative dossiers)
    all_evals_df = pd.read_sql(
        'SELECT week_id as forecast_week, item_id as unique_id, h1_mape as mape FROM evaluations',
        conn,
    )

    # Clean slate
    conn.execute('DELETE FROM narratives')
    conn.commit()

    # ── Week and item narratives ──────────────────────────────────────────────
    print()
    for forecast_week in tqdm(bad_weeks, desc='Week narratives'):
        xai_rows = load_xai(conn, forecast_week)
        shap_rows = [r for r in xai_rows if r['xai_type'] == 'shap']
        if not shap_rows:
            continue

        week_evals = all_evals_df[all_evals_df['forecast_week'] == forecast_week]
        n_items    = int(conn.execute(
            'SELECT COUNT(DISTINCT item_id) FROM evaluations WHERE week_id=?', (forecast_week,)
        ).fetchone()[0])

        week_doss = build_week_dossier(
            forecast_week, shap_rows, zscore_map.get(forecast_week), n_items
        )
        week_narr = narrator.generate(WEEK_NARRATIVE_PROMPT, week_doss)
        if week_narr:
            insert_narrative(conn, 'week', forecast_week, week_narr, narrator.model_id)

        # Item narratives for the TOP_ITEMS worst SKUs
        top_items = (
            week_evals.nlargest(TOP_ITEMS, 'mape')['unique_id'].tolist()
        )
        shap_by_item = {r['item_id']: r for r in shap_rows}
        cf_rows   = [r for r in xai_rows if r['xai_type'] == 'counterfactual']
        cont_rows = [r for r in xai_rows if r['xai_type'] == 'contrastive']
        cf_by_item   = {r['item_id']: r for r in cf_rows}
        cont_by_item = {r['item_id']: r for r in cont_rows}

        for item_id in top_items:
            shap_p = json.loads(shap_by_item[item_id]['payload']) if item_id in shap_by_item else None
            cf_p   = json.loads(cf_by_item[item_id]['payload'])   if item_id in cf_by_item   else None
            cont_p = json.loads(cont_by_item[item_id]['payload']) if item_id in cont_by_item else None
            item_doss = build_item_dossier(forecast_week, item_id, shap_p, cf_p, cont_p)
            item_narr = narrator.generate(ITEM_NARRATIVE_PROMPT, item_doss)
            if item_narr:
                insert_narrative(conn, 'item', f'{forecast_week}::{item_id}', item_narr, narrator.model_id)

    # ── Executive synthesis ───────────────────────────────────────────────────
    print('\nGenerating executive narrative...')
    n_total = conn.execute(
        'SELECT COUNT(DISTINCT week_id) FROM evaluations'
    ).fetchone()[0]
    drivers_list = compute_recurring_drivers(load_all_shap_payloads(conn))
    exec_doss = build_executive_dossier(drivers_list, len(bad_weeks), n_total)
    exec_narr = narrator.generate(EXECUTIVE_NARRATIVE_PROMPT, exec_doss)
    if exec_narr:
        insert_narrative(conn, 'executive', 'overall', exec_narr, narrator.model_id)

    saved = conn.execute('SELECT COUNT(*) FROM narratives').fetchone()[0]
    conn.close()
    print(f'\nDone — {saved} narratives saved -> {DB_PATH}')
    print('Next: uv run python data_quality.py')


if __name__ == '__main__':
    main()
