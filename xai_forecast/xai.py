"""
XAI payloads: SHAP, counterfactual, contrastive.

All payload functions return list[dict] with keys: week_id, item_id, xai_type, payload (JSON str).

shap_payloads additionally returns a shap_cache {uid: 1-D shap array} so contrastive can
reuse bad-item SHAP values without recomputing them.
"""

import json
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb

from xai_forecast.features import FEATURE_COLS

_CF_PERTURBATIONS = [
    ('no_snap',         {'snap': 0}),
    ('no_event',        {'has_event': 0, 'event_type_enc': 0}),
    ('no_price_change', {'price_change_pct': 0}),
]


def make_explainer(model: lgb.LGBMRegressor) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def shap_payloads(
    explainer: shap.TreeExplainer,
    model: lgb.LGBMRegressor,
    week_df: pd.DataFrame,
    forecast_week: str,
    items: list[str],
    actual_map: dict[str, float],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """
    Returns (db_rows, shap_cache).

    shap_cache maps uid -> full 1-D SHAP array (len = FEATURE_COLS) in log-margin space.
    Pass shap_cache to contrastive_payloads to avoid recomputing bad-item SHAP there.

    Payload includes other_features_shap (sum of non-top-5 contributions) so the waterfall
    in app.py can show an honest residual bar that reconciles to the actual log-prediction:
        base_value_log + Σ(top5 shap) + other_features_shap ≈ log(prediction)
    """
    rows = week_df[week_df['unique_id'].isin(items)][['unique_id'] + FEATURE_COLS].fillna(0)
    if rows.empty:
        return [], {}

    X = rows[FEATURE_COLS]
    sv = explainer.shap_values(X)
    # Tweedie uses log-link: base_value and shap_values are in log-margin space.
    # base_log + sum(shap_values) = log(prediction). Feature ranking by |shap| is correct.
    base_log = float(explainer.expected_value)
    preds = model.predict(X).clip(min=0)

    results = []
    shap_cache: dict[str, np.ndarray] = {}
    for i, uid in enumerate(rows['unique_id']):
        shap_cache[uid] = sv[i]
        top_idx = np.argsort(np.abs(sv[i]))[::-1][:5]
        other_idx = np.argsort(np.abs(sv[i]))[::-1][5:]
        other_shap = float(np.sum(sv[i][other_idx]))

        actual = actual_map.get(uid, np.nan)
        results.append({
            'week_id': forecast_week,
            'item_id': uid,
            'xai_type': 'shap',
            'payload': json.dumps({
                'base_value_log': round(base_log, 4),
                'prediction': round(float(preds[i]), 4),
                'actual': round(float(actual), 4) if not np.isnan(actual) else None,
                'error_pct': round(abs(float(preds[i]) - actual) / actual * 100, 2) if actual > 0 else None,
                'shap_note': 'values in log-margin space (Tweedie log-link); ranking by |shap| is valid',
                'other_features_shap': round(other_shap, 4),
                'top_features': [
                    {
                        'feature': FEATURE_COLS[j],
                        'shap_value': round(float(sv[i][j]), 4),
                        'feature_value': round(float(rows[FEATURE_COLS].iloc[i, j]), 4),
                    }
                    for j in top_idx
                ],
            }),
        })
    return results, shap_cache


def counterfactual_payloads(
    model: lgb.LGBMRegressor,
    week_df: pd.DataFrame,
    forecast_week: str,
    items: list[str],
    actual_map: dict[str, float],
) -> list[dict]:
    rows = week_df[week_df['unique_id'].isin(items)][['unique_id'] + FEATURE_COLS].fillna(0).reset_index(drop=True)
    if rows.empty:
        return []

    X_orig = rows[FEATURE_COLS].copy()
    preds_orig = model.predict(X_orig).clip(min=0)

    cf_preds: dict[str, np.ndarray] = {}
    for scenario, overrides in _CF_PERTURBATIONS:
        X_cf = X_orig.copy()
        for col, val in overrides.items():
            if col in X_cf.columns:
                X_cf[col] = val
        cf_preds[scenario] = model.predict(X_cf).clip(min=0)

    results = []
    for i, uid in enumerate(rows['unique_id']):
        actual = actual_map.get(uid, np.nan)
        orig = float(preds_orig[i])
        results.append({
            'week_id': forecast_week,
            'item_id': uid,
            'xai_type': 'counterfactual',
            'payload': json.dumps({
                'prediction_original': round(orig, 4),
                'actual': round(float(actual), 4) if not np.isnan(actual) else None,
                'scenarios': [
                    {
                        'scenario': scenario,
                        # was_active: True if any overridden feature was non-zero for this SKU
                        'was_active': any(float(X_orig.iloc[i][col]) != 0 for col in overrides),
                        'prediction_cf': round(float(cf_preds[scenario][i]), 4),
                        'delta': round(float(cf_preds[scenario][i]) - orig, 4),
                        'delta_pct': round(
                            (float(cf_preds[scenario][i]) - orig) / orig * 100 if orig > 0 else 0, 2
                        ),
                    }
                    for scenario, overrides in _CF_PERTURBATIONS
                ],
            }),
        })
    return results


def contrastive_payloads(
    explainer: shap.TreeExplainer,
    week_df: pd.DataFrame,
    forecast_week: str,
    items: list[str],
    all_evals: pd.DataFrame,
    conn: sqlite3.Connection,
    bad_shap_cache: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """
    For each bad item find a good reference week (same ISO week-of-year, MAPE < 15%)
    and diff the SHAP profiles.

    Items with no same-WOY good week are skipped — no fallback to a different-seasonality
    week (that would break the "similar context" claim in the explanation).

    Reference features are loaded once per unique ref week (not once per item).
    Bad-item SHAP from shap_payloads is reused via bad_shap_cache when available,
    avoiding a second explainer call per item.
    """
    from xai_forecast.db import load_features_week

    bad_woy = pd.Timestamp(forecast_week).isocalendar()[1]

    # ── Resolve ref week for each item (same WOY only) ──────────────────────
    item_ref: dict[str, tuple[str, float]] = {}  # uid -> (ref_forecast_week, good_mape)
    for uid in items:
        good_weeks = all_evals[
            (all_evals['unique_id'] == uid) & (all_evals['mape'] < 15)
        ]
        if good_weeks.empty:
            continue
        same_woy = good_weeks[
            good_weeks['forecast_week'].apply(
                lambda w: pd.Timestamp(w).isocalendar()[1] == bad_woy
            )
        ]
        if same_woy.empty:
            continue  # no same-WOY good week — skip rather than use different seasonality
        ref_row = same_woy.iloc[-1]
        item_ref[uid] = (ref_row['forecast_week'], float(ref_row['mape']))

    if not item_ref:
        return []

    # ── Load ref features once per unique ref week ───────────────────────────
    ref_week_cache: dict[str, pd.DataFrame] = {}
    for ref_fw in set(rw for rw, _ in item_ref.values()):
        ref_week_cache[ref_fw] = load_features_week(conn, ref_fw)

    # ── Bad-item SHAP: reuse cache or compute once for all items ─────────────
    if bad_shap_cache is not None:
        bad_shap_by_uid = {uid: bad_shap_cache[uid] for uid in item_ref if uid in bad_shap_cache}
    else:
        bad_rows = (
            week_df[week_df['unique_id'].isin(item_ref.keys())]
            [['unique_id'] + FEATURE_COLS].fillna(0).reset_index(drop=True)
        )
        if bad_rows.empty:
            return []
        sv_bad_all = explainer.shap_values(bad_rows[FEATURE_COLS])
        bad_shap_by_uid = {uid: sv_bad_all[i] for i, uid in enumerate(bad_rows['unique_id'])}

    # ── Ref-item SHAP: batch by ref week ─────────────────────────────────────
    items_by_ref: dict[str, list[str]] = defaultdict(list)
    for uid, (ref_fw, _) in item_ref.items():
        items_by_ref[ref_fw].append(uid)

    ref_shap_by_uid: dict[str, np.ndarray] = {}
    for ref_fw, ref_uids in items_by_ref.items():
        ref_df = ref_week_cache[ref_fw]
        ref_items_df = (
            ref_df[ref_df['unique_id'].isin(ref_uids)]
            [['unique_id'] + FEATURE_COLS].fillna(0).reset_index(drop=True)
        )
        if ref_items_df.empty:
            continue
        sv_ref = explainer.shap_values(ref_items_df[FEATURE_COLS])
        for i, uid in enumerate(ref_items_df['unique_id']):
            ref_shap_by_uid[uid] = sv_ref[i]

    # ── Build payloads ────────────────────────────────────────────────────────
    results = []
    for uid, (ref_fw, good_mape) in item_ref.items():
        if uid not in bad_shap_by_uid or uid not in ref_shap_by_uid:
            continue

        bad_item = week_df[week_df['unique_id'] == uid]
        ref_item = ref_week_cache[ref_fw][ref_week_cache[ref_fw]['unique_id'] == uid]
        if bad_item.empty or ref_item.empty:
            continue

        sv_bad = bad_shap_by_uid[uid]
        sv_ref_item = ref_shap_by_uid[uid]

        diffs = sorted(
            [
                {
                    'feature': FEATURE_COLS[j],
                    'shap_diff': round(float(sv_bad[j] - sv_ref_item[j]), 4),
                    'bad_value': round(float(bad_item[FEATURE_COLS].iloc[0, j]), 4),
                    'good_value': round(float(ref_item[FEATURE_COLS].iloc[0, j]), 4),
                    'bad_shap': round(float(sv_bad[j]), 4),
                    'good_shap': round(float(sv_ref_item[j]), 4),
                }
                for j in range(len(FEATURE_COLS))
            ],
            key=lambda d: abs(d['shap_diff']),
            reverse=True,
        )

        results.append({
            'week_id': forecast_week,
            'item_id': uid,
            'xai_type': 'contrastive',
            'payload': json.dumps({
                'bad_week': forecast_week,
                'good_week': ref_fw,
                'good_week_mape': round(good_mape, 2),
                'seasonality_matched': True,
                'top_diffs': diffs[:5],
            }),
        })
    return results
