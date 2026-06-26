"""
Smoke test: train once, then run 5 forecast weeks in parallel (concurrency=5).
Requires ingest.py to have run first.

Usage:
    uv run python smoke_test.py
"""

import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from xai_forecast.db import (
    get_conn, get_all_weeks, load_raw_window,
    insert_forecasts, insert_evaluations, insert_xai, week_summary,
)
from xai_forecast.features import compute_features, FEATURE_COLS, HISTORY_BUFFER
from xai_forecast.train import train_model
from xai_forecast.forecast import make_forecasts
from xai_forecast.evaluate import evaluate_h1
from xai_forecast.xai import make_explainer, shap_payloads, counterfactual_payloads

TRAIN_WINDOW  = 156
WEEKS_TO_TEST = 5
CONCURRENCY   = 5
TOP_N         = 5
DB_PATH       = 'db/forecasting.db'

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg)


def fail(msg: str) -> None:
    print(f'  [FAIL] {msg}')
    sys.exit(1)


# ── Per-week worker (runs in thread pool) ─────────────────────────────────────

def run_week(forecast_week: str, weeks: list, model, explainer) -> dict:
    t0 = time.perf_counter()
    step = weeks.index(forecast_week)

    # Each thread gets its own SQLite connection (WAL mode allows concurrent reads)
    conn = get_conn(DB_PATH)
    buf_start = weeks[max(0, step - HISTORY_BUFFER)]
    raw       = load_raw_window(conn, buf_start, forecast_week)
    conn.close()

    t_sql = time.perf_counter() - t0

    feat    = compute_features(raw)
    week_df = feat[feat['week'] == forecast_week]

    t_feat = time.perf_counter() - t0 - t_sql

    preds   = make_forecasts(model, week_df, forecast_week)
    eval_df = evaluate_h1(preds, week_df[['unique_id', 'y']])
    eval_df['forecast_week'] = forecast_week

    top_items  = eval_df.nlargest(TOP_N, 'mape')['unique_id'].tolist()
    actual_map = dict(zip(eval_df['unique_id'], eval_df['actual']))
    shap_rows  = shap_payloads(explainer, model, week_df, forecast_week, top_items, actual_map)
    cf_rows    = counterfactual_payloads(model, week_df, forecast_week, top_items, actual_map)

    t_total = time.perf_counter() - t0

    log(f'  [done] {forecast_week}  sql={t_sql:.1f}s  feat={t_feat:.1f}s  total={t_total:.1f}s'
        f'  MAPE avg={eval_df["mape"].mean():.1f}%  median={eval_df["mape"].median():.1f}%'
        f'  shap={len(shap_rows)}  cf={len(cf_rows)}')

    return {
        'forecast_week': forecast_week,
        'eval_df':       eval_df,
        'preds':         preds,
        'shap_rows':     shap_rows,
        'cf_rows':       cf_rows,
        'timings':       {'sql': t_sql, 'feat': t_feat, 'total': t_total},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    wall_start = time.perf_counter()

    print('=' * 60)
    print('Smoke test  (5 weeks, concurrency=5)')
    print('=' * 60)

    # ── Setup ─────────────────────────────────────────────────────
    t = time.perf_counter()
    conn  = get_conn(DB_PATH)
    weeks = get_all_weeks(conn)
    conn.close()

    if not weeks:
        fail('No data -- run: uv run python ingest.py')
    if len(weeks) <= TRAIN_WINDOW + WEEKS_TO_TEST:
        fail(f'Not enough weeks: need {TRAIN_WINDOW + WEEKS_TO_TEST}, have {len(weeks)}')
    print(f'\n[STEP] Setup  ({time.perf_counter()-t:.2f}s)')
    print(f'  {len(weeks)} weeks in DB ({weeks[0]} -> {weeks[-1]})')

    # ── Train (single model, weeks[TRAIN_WINDOW] cutoff) ──────────
    t = time.perf_counter()
    conn  = get_conn(DB_PATH)
    cutoff       = weeks[TRAIN_WINDOW]
    window_start = weeks[0]
    buffer_start = weeks[0]   # max(0, 156-156-52) = 0
    raw_df       = load_raw_window(conn, buffer_start, cutoff)
    conn.close()

    features_df = compute_features(raw_df)
    train_df    = features_df[features_df['week'] > window_start].dropna(subset=FEATURE_COLS)

    if len(train_df) == 0:
        fail('Empty training set')

    model     = train_model(train_df)
    explainer = make_explainer(model)
    print(f'\n[STEP] Train LightGBM  ({time.perf_counter()-t:.2f}s)')
    print(f'  {len(train_df):,} training rows | cutoff={cutoff}')
    missing = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing:
        fail(f'Missing feature cols: {missing}')
    print(f'  Features OK ({len(FEATURE_COLS)} cols)')

    # ── Pick 5 forecast weeks spread across backtest range ─────────
    # Space them evenly so we sample early, mid, late in the backtest
    backtest_start = TRAIN_WINDOW + 1
    step_size      = max(1, (len(weeks) - backtest_start - 1) // (WEEKS_TO_TEST - 1))
    forecast_weeks = [weeks[backtest_start + i * step_size] for i in range(WEEKS_TO_TEST)]

    print(f'\n[STEP] Parallel forecast ({WEEKS_TO_TEST} weeks, concurrency={CONCURRENCY})')
    print(f'  Weeks: {forecast_weeks}')
    print(f'  Sequential estimate: ~{WEEKS_TO_TEST * 36:.0f}s  |  parallel target: ~36s\n')

    t_parallel = time.perf_counter()
    results = {}

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(run_week, fw, weeks, model, explainer): fw
            for fw in forecast_weeks
        }
        for future in as_completed(futures):
            fw = futures[future]
            try:
                results[fw] = future.result()
            except Exception as exc:
                fail(f'{fw} raised {exc}')

    t_parallel_total = time.perf_counter() - t_parallel

    # ── Validate results ───────────────────────────────────────────
    print(f'\n[STEP] Validation')
    all_ok = True
    for fw in forecast_weeks:
        r = results[fw]
        ok_shap  = len(r['shap_rows']) == TOP_N
        ok_cf    = len(r['cf_rows'])   == TOP_N
        ok_key   = all(x['week_id'] == fw for x in r['shap_rows'] + r['cf_rows'])
        ok_mape  = (r['eval_df']['mape'] >= 0).all()
        ok_preds = (r['preds']['h1'] >= 0).all()
        status   = '[OK]  ' if all([ok_shap, ok_cf, ok_key, ok_mape, ok_preds]) else '[FAIL]'
        if status == '[FAIL]':
            all_ok = False
        print(f'  {status} {fw}  shap={ok_shap}  cf={ok_cf}  key_match={ok_key}  mape_ok={ok_mape}  preds_ok={ok_preds}')

    if not all_ok:
        fail('One or more weeks failed validation')

    # ── Write to SQLite (serial — avoid write contention) ──────────
    t = time.perf_counter()
    conn = get_conn(DB_PATH)
    for r in results.values():
        fw = r['forecast_week']
        insert_forecasts(conn, [
            {'week_id': fw, 'item_id': row['unique_id'],
             'h1': row['h1'], 'trained_at': 'smoke-test'}
            for row in r['preds'].to_dict('records')
        ])
        insert_evaluations(conn, [
            {'week_id': fw, 'item_id': row['unique_id'],
             'h1_mape': row['mape'], 'h1_mae': row['mae'],
             'is_bad_week': 0, 'mape_zscore': 0.0}
            for row in r['eval_df'].to_dict('records')
        ])
        insert_xai(conn, r['shap_rows'] + r['cf_rows'])

    summary = week_summary(conn)
    conn.close()
    print(f'\n[STEP] SQLite write + readback  ({time.perf_counter()-t:.2f}s)')
    print(f'  {len(summary)} weeks written | avg MAPE across weeks: '
          f'{summary["avg_mape"].mean():.1f}%')

    # ── Timing summary ─────────────────────────────────────────────
    timings = [r['timings'] for r in results.values()]
    print(f'\n{"=" * 60}')
    print(f'Timing summary')
    print(f'{"=" * 60}')
    print(f'  Parallel wall time:      {t_parallel_total:.1f}s  ({WEEKS_TO_TEST} weeks @ concurrency={CONCURRENCY})')
    print(f'  Sequential equivalent:   ~{sum(t["total"] for t in timings):.1f}s')
    print(f'  Speedup:                 {sum(t["total"] for t in timings) / t_parallel_total:.1f}x')
    print(f'  Per-week avg (total):    {sum(t["total"] for t in timings)/len(timings):.1f}s')
    print(f'  Per-week avg (sql):      {sum(t["sql"] for t in timings)/len(timings):.1f}s')
    print(f'  Per-week avg (features): {sum(t["feat"] for t in timings)/len(timings):.1f}s')
    print(f'  Total wall (incl train): {time.perf_counter()-wall_start:.1f}s')
    print(f'\nAll checks passed.')


if __name__ == '__main__':
    main()
