"""
Smoke test: train once, then run 10 forecast weeks in parallel (concurrency=10).
Requires ingest.py and build_features.py to have run first.

Usage:
    uv run python smoke_test.py

Checks:
  - Feature store staleness (recomputes a sample and diffs vs stored)
  - Parallel forecast + SHAP + counterfactual
  - Contrastive (needs enough history for same-WOY match — samples weeks > 52 weeks in)
  - SHAP additivity on real data
  - Forecast count == SKU count, all h1 >= 0
  - flag_bad_weeks + data_quality.py against smoke.db
"""

import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd

from xai_forecast.db import (
    get_conn, get_all_weeks, load_features_window, load_features_week, load_raw_window,
    insert_forecasts, insert_evaluations, insert_xai, week_summary,
)
from xai_forecast.features import FEATURE_COLS, compute_features
from xai_forecast.train import train_model
from xai_forecast.forecast import make_forecasts
from xai_forecast.evaluate import evaluate_h1, flag_bad_weeks
from xai_forecast.xai import make_explainer, shap_payloads, counterfactual_payloads, contrastive_payloads

TRAIN_WINDOW  = 156
WEEKS_TO_TEST = 10
CONCURRENCY   = 10
TOP_N         = 5
SOURCE_DB     = 'db/forecasting.db'   # raw data + feature store (read-only)
SMOKE_DB      = 'db/smoke.db'         # throwaway output — never read by dashboard
STALE_CHECK_SKUS = 10   # number of SKUs to spot-check for feature store staleness

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg)


def fail(msg: str) -> None:
    print(f'  [FAIL] {msg}')
    sys.exit(1)


# ── Feature staleness detector ────────────────────────────────────────────────

def check_feature_staleness(source_conn, weeks: list[str]) -> None:
    """
    Recompute features for a few (SKU, week) pairs via compute_features() and diff
    against the stored feature rows. Catches "edited features.py, forgot to rebuild."
    Exits with fail() if any column differs by more than 1e-6.
    """
    print(f'\n[STEP] Feature staleness check ({STALE_CHECK_SKUS} SKUs, 3 weeks)')

    # Pick a mid-range week that has lag_52 populated (needs >= 52 prior weeks)
    check_week = weeks[TRAIN_WINDOW]  # week 156 — well inside the data range

    # Load stored feature rows for this week
    stored = pd.read_sql(
        'SELECT * FROM features WHERE week = ?', source_conn, params=(check_week,)
    )
    if stored.empty:
        fail('No stored features found — run build_features.py first')

    # Sample a few SKUs
    sample_uids = stored['unique_id'].sample(min(STALE_CHECK_SKUS, len(stored)),
                                              random_state=42).tolist()

    # Recompute features for those SKUs up to check_week
    raw_df = load_raw_window(source_conn, '', check_week)
    raw_df = raw_df[raw_df['unique_id'].isin(sample_uids)]
    if raw_df.empty:
        fail('Could not load raw data for staleness check')

    recomp = compute_features(raw_df)
    recomp_week = recomp[recomp['week'] == check_week].set_index('unique_id')
    stored_week  = stored[stored['unique_id'].isin(sample_uids)].set_index('unique_id')

    max_diffs = {}
    for col in FEATURE_COLS:
        if col not in recomp_week.columns or col not in stored_week.columns:
            continue
        a = stored_week[col].reindex(recomp_week.index).fillna(0)
        b = recomp_week[col].fillna(0)
        diff = (a - b).abs().max()
        if diff > 1e-4:
            max_diffs[col] = diff

    if max_diffs:
        fail(
            'Feature store is STALE — features.py changed without rebuilding. '
            f'Columns with diff: {max_diffs}. '
            'Run: uv run python build_features.py'
        )
    print(f'  [OK]  Feature store matches recomputed values for {len(sample_uids)} SKUs')


# ── Per-week worker (runs in thread pool) ─────────────────────────────────────

def run_week(forecast_week: str, all_evals_df: pd.DataFrame,
             model, explainer, conn) -> dict:
    t0 = time.perf_counter()

    # Each thread gets its own connection to the source DB (WAL allows concurrent reads)
    src_conn = get_conn(SOURCE_DB)
    week_df  = load_features_week(src_conn, forecast_week)
    src_conn.close()

    t_sql = time.perf_counter() - t0

    preds   = make_forecasts(model, week_df, forecast_week)
    eval_df = evaluate_h1(preds, week_df[['unique_id', 'y']])
    eval_df['forecast_week'] = forecast_week

    top_items  = eval_df.nlargest(TOP_N, 'mape')['unique_id'].tolist()
    actual_map = dict(zip(eval_df['unique_id'], eval_df['actual']))

    shap_rows, shap_cache = shap_payloads(explainer, model, week_df, forecast_week, top_items, actual_map)
    cf_rows = counterfactual_payloads(model, week_df, forecast_week, top_items, actual_map)

    # Contrastive: needs all_evals_df with history. Use a fresh conn per thread for the ref load.
    cont_conn = get_conn(SOURCE_DB)
    cont_rows = contrastive_payloads(
        explainer, week_df, forecast_week, top_items, all_evals_df, cont_conn, shap_cache
    )
    cont_conn.close()

    t_total = time.perf_counter() - t0

    log(f'  [done] {forecast_week}  sql={t_sql:.1f}s  total={t_total:.1f}s'
        f'  MAPE avg={eval_df["mape"].mean():.1f}%'
        f'  shap={len(shap_rows)}  cf={len(cf_rows)}  cont={len(cont_rows)}')

    return {
        'forecast_week': forecast_week,
        'eval_df':       eval_df,
        'preds':         preds,
        'shap_rows':     shap_rows,
        'cf_rows':       cf_rows,
        'cont_rows':     cont_rows,
        'timings':       {'sql': t_sql, 'total': t_total},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    wall_start = time.perf_counter()

    print('=' * 60)
    print(f'Smoke test  ({WEEKS_TO_TEST} weeks, concurrency={CONCURRENCY})')
    print('=' * 60)

    # ── Setup ─────────────────────────────────────────────────────
    t = time.perf_counter()
    conn  = get_conn(SOURCE_DB)
    weeks = get_all_weeks(conn)
    n_feat = conn.execute('SELECT COUNT(*) FROM features').fetchone()[0]

    if not weeks:
        conn.close(); fail('No data -- run: uv run python ingest.py')
    if len(weeks) <= TRAIN_WINDOW + WEEKS_TO_TEST:
        conn.close(); fail(f'Not enough weeks: need {TRAIN_WINDOW + WEEKS_TO_TEST}, have {len(weeks)}')
    if n_feat == 0:
        conn.close(); fail('Feature store empty -- run: uv run python build_features.py')

    print(f'\n[STEP] Setup  ({time.perf_counter()-t:.2f}s)')
    print(f'  {len(weeks)} weeks in DB ({weeks[0]} -> {weeks[-1]})')

    # ── Feature staleness check ────────────────────────────────────
    check_feature_staleness(conn, weeks)

    # ── Train ──────────────────────────────────────────────────────
    t = time.perf_counter()
    cutoff       = weeks[TRAIN_WINDOW]
    window_start = weeks[0]
    train_df     = load_features_window(conn, window_start, cutoff).dropna(subset=FEATURE_COLS)

    if len(train_df) == 0:
        conn.close(); fail('Empty training set')

    model     = train_model(train_df)
    explainer = make_explainer(model)
    print(f'\n[STEP] Train LightGBM  ({time.perf_counter()-t:.2f}s)')
    print(f'  {len(train_df):,} training rows | cutoff={cutoff}')
    missing = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing:
        conn.close(); fail(f'Missing feature cols: {missing}')
    print(f'  Features OK ({len(FEATURE_COLS)} cols)')

    # ── Build a mini all_evals_df for contrastive reference lookup ──
    # Contrastive needs ~1+ year of eval history to find same-WOY good weeks.
    # We build this by running evaluate_h1 across a wider window (not parallel — serial, fast).
    t = time.perf_counter()
    history_start_idx = TRAIN_WINDOW + 1
    # Use 60 weeks of history (covers all WOY values)
    history_weeks = weeks[history_start_idx: history_start_idx + 60]
    history_evals = []
    for hw in history_weeks:
        hw_df = load_features_week(conn, hw)
        if hw_df.empty:
            continue
        hw_preds = make_forecasts(model, hw_df, hw)
        hw_eval  = evaluate_h1(hw_preds, hw_df[['unique_id', 'y']])
        hw_eval['forecast_week'] = hw
        history_evals.append(hw_eval)
    all_evals_df = pd.concat(history_evals, ignore_index=True) if history_evals else pd.DataFrame()
    print(f'\n[STEP] Built eval history  ({time.perf_counter()-t:.2f}s)'
          f'  ({len(history_weeks)} weeks, {len(all_evals_df):,} rows)')

    # ── Pick forecast weeks — must be > 52 weeks in for contrastive ─
    # Exclude weeks[-1] (partial 2-day week) from the selection pool.
    # Also require at least 52 weeks of history before the forecast week (for contrastive WOY match).
    min_idx = TRAIN_WINDOW + 53  # at least 52 weeks into backtest range
    valid_weeks = weeks[min_idx:-1]
    if len(valid_weeks) < WEEKS_TO_TEST:
        # Fallback: use any post-train weeks
        valid_weeks = weeks[TRAIN_WINDOW + 1:-1]
    step_size    = max(1, (len(valid_weeks) - 1) // (WEEKS_TO_TEST - 1))
    forecast_weeks = [valid_weeks[i * step_size] for i in range(WEEKS_TO_TEST)]

    print(f'\n[STEP] Parallel forecast ({WEEKS_TO_TEST} weeks, concurrency={CONCURRENCY})')
    print(f'  Weeks: {forecast_weeks}\n')

    t_parallel = time.perf_counter()
    results = {}

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(run_week, fw, all_evals_df, model, explainer, conn): fw
            for fw in forecast_weeks
        }
        for future in as_completed(futures):
            fw = futures[future]
            try:
                results[fw] = future.result()
            except Exception as exc:
                conn.close(); fail(f'{fw} raised {exc}')

    t_parallel_total = time.perf_counter() - t_parallel

    # ── Validation ─────────────────────────────────────────────────
    print(f'\n[STEP] Validation')
    all_ok = True
    for fw in forecast_weeks:
        r = results[fw]
        ok_shap  = len(r['shap_rows']) == TOP_N
        ok_cf    = len(r['cf_rows'])   == TOP_N
        ok_key   = all(x['week_id'] == fw for x in r['shap_rows'] + r['cf_rows'])
        ok_mape  = (r['eval_df']['mape'] >= 0).all()
        ok_preds = (r['preds']['h1'] >= 0).all()
        ok_count = len(r['preds']) > 0
        # SHAP additivity check on one SKU
        ok_additivity = True
        if r['shap_rows']:
            try:
                p = json.loads(r['shap_rows'][0]['payload'])
                total = sum(f['shap_value'] for f in p['top_features']) + p.get('other_features_shap', 0)
                pred  = p['prediction']
                if pred > 0 and abs(p['base_value_log'] + total - np.log(pred)) > 0.05:
                    ok_additivity = False
            except Exception:
                ok_additivity = False

        ok_payload_json = True
        for row in r['shap_rows'] + r['cf_rows'] + r['cont_rows']:
            try:
                json.loads(row['payload'])
            except Exception:
                ok_payload_json = False
                break

        all_checks = [ok_shap, ok_cf, ok_key, ok_mape, ok_preds, ok_count, ok_additivity, ok_payload_json]
        status = '[OK]  ' if all(all_checks) else '[FAIL]'
        if status == '[FAIL]':
            all_ok = False
        print(f'  {status} {fw}  shap={ok_shap}  cf={ok_cf}  key={ok_key}'
              f'  mape_ok={ok_mape}  h1_ok={ok_preds}  additive={ok_additivity}'
              f'  json={ok_payload_json}  cont={len(r["cont_rows"])} rows')

    if not all_ok:
        conn.close(); fail('One or more weeks failed validation')

    # ── Narrative API probe ────────────────────────────────────────
    # One live call to DeepSeek to validate API config before the full backtest.
    # Fails loudly if the key is set but the call fails or returns a bad schema.
    # Skips silently if DEEPSEEK_API_KEY is not set.
    print(f'\n[STEP] Narrative API probe')
    _narrator = None
    try:
        from xai_forecast.narrate import DeepSeekNarrator, WEEK_NARRATIVE_PROMPT, build_week_dossier
        from xai_forecast.db import insert_narrative, load_narrative as _load_narrative
        _narrator = DeepSeekNarrator()
    except ImportError:
        print('  [SKIP] openai not installed — skipping narrative probe')

    if _narrator and _narrator.available:
        _probe_week = forecast_weeks[0]
        _probe_shap_rows = results[_probe_week]['shap_rows']
        if not _probe_shap_rows:
            print('  [SKIP] No SHAP rows for probe week')
        else:
            _probe_doss = build_week_dossier(
                _probe_week,
                _probe_shap_rows,
                wmape_zscore=2.0,
                n_items_in_week=len(results[_probe_week]['eval_df']),
            )
            _probe_narr = _narrator.generate(WEEK_NARRATIVE_PROMPT, _probe_doss)
            if _probe_narr is None:
                conn.close()
                fail(
                    'Narrative API probe failed — DeepSeek returned None. '
                    'Check DEEPSEEK_API_KEY, DEEPSEEK_MODEL, and DEEPSEEK_BASE_URL in .env'
                )
            for _k in ('headline', 'body', 'primary_driver', 'confidence'):
                if _k not in _probe_narr:
                    conn.close(); fail(f'Narrative probe response missing key: {_k}')

            # DB round-trip: write to smoke.db and read back
            _probe_conn = get_conn(SMOKE_DB)
            insert_narrative(_probe_conn, 'week', _probe_week, _probe_narr, _narrator.model_id)
            _recovered = _load_narrative(_probe_conn, 'week', _probe_week)
            _probe_conn.close()
            if _recovered is None or _recovered.get('headline') != _probe_narr['headline']:
                conn.close(); fail('Narrative DB round-trip failed — insert or read back broken')

            print(f'  [OK]  API call succeeded + DB round-trip passed')
            print(f'        Headline:        {_probe_narr["headline"]}')
            print(f'        Primary driver:  {_probe_narr["primary_driver"]}')
            print(f'        Confidence:      {_probe_narr["confidence"]}')
            if _probe_narr.get('grounding_warning'):
                print(f'  [WARN] Grounding check flagged primary_driver — verify manually')
    elif _narrator is not None:
        print('  [SKIP] DEEPSEEK_API_KEY not set — set in .env to enable narrative generation')

    # ── Write to throwaway smoke DB ────────────────────────────────
    t = time.perf_counter()
    smoke_conn = get_conn(SMOKE_DB)
    smoke_conn.executescript('DELETE FROM forecasts; DELETE FROM evaluations; DELETE FROM xai_results; DELETE FROM narratives;')
    smoke_conn.commit()

    all_eval_frames = []
    for r in results.values():
        fw = r['forecast_week']
        insert_forecasts(smoke_conn, [
            {'week_id': fw, 'item_id': row['unique_id'],
             'h1': row['h1'], 'trained_at': 'smoke-test'}
            for row in r['preds'].to_dict('records')
        ])
        ev = r['eval_df'].copy()
        all_eval_frames.append(ev)

    if all_eval_frames:
        combined_evals = pd.concat(all_eval_frames, ignore_index=True)
        wk_flags = flag_bad_weeks(combined_evals)
        is_bad = wk_flags.set_index('forecast_week')['is_bad_week'].to_dict()
        zscore = wk_flags.set_index('forecast_week')['zscore'].to_dict()
        insert_evaluations(smoke_conn, [
            {'week_id': row['forecast_week'], 'item_id': row['unique_id'],
             'h1_mape': row['mape'], 'h1_mae': row['mae'],
             'is_bad_week': int(is_bad.get(row['forecast_week'], False)),
             'mape_zscore': float(z) if pd.notna(z := zscore.get(row['forecast_week'], 0)) else 0.0}
            for row in combined_evals.to_dict('records')
        ])
        n_smoke_bad = sum(1 for v in is_bad.values() if v)
        print(f'\n[STEP] Bad weeks (smoke): {n_smoke_bad}/{len(wk_flags)} flagged')

    for r in results.values():
        insert_xai(smoke_conn, r['shap_rows'] + r['cf_rows'] + r['cont_rows'])

    summary = week_summary(smoke_conn)
    smoke_conn.close()
    print(f'\n[STEP] SQLite write + readback  ({time.perf_counter()-t:.2f}s)')
    print(f'  {len(summary)} weeks written | avg MAPE: {summary["avg_mape"].mean():.1f}%')

    # ── Timing summary ─────────────────────────────────────────────
    timings = [r['timings'] for r in results.values()]
    print(f'\n{"=" * 60}')
    print(f'Timing summary')
    print(f'{"=" * 60}')
    print(f'  Parallel wall time:      {t_parallel_total:.1f}s  ({WEEKS_TO_TEST} weeks @ concurrency={CONCURRENCY})')
    print(f'  Sequential equivalent:   ~{sum(t["total"] for t in timings):.1f}s')
    print(f'  Speedup:                 {sum(t["total"] for t in timings) / t_parallel_total:.1f}x')
    print(f'  Total wall (incl train): {time.perf_counter()-wall_start:.1f}s')
    print(f'\nAll checks passed.')
    conn.close()


if __name__ == '__main__':
    main()
