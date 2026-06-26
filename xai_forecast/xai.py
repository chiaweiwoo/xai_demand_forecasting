import json
import sqlite3
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
) -> list[dict]:
    rows = week_df[week_df['unique_id'].isin(items)][['unique_id'] + FEATURE_COLS].fillna(0)
    if rows.empty:
        return []

    X = rows[FEATURE_COLS].values
    sv = explainer.shap_values(X)
    base = float(explainer.expected_value)
    preds = model.predict(X).clip(min=0)

    results = []
    for i, uid in enumerate(rows['unique_id']):
        top_idx = np.argsort(np.abs(sv[i]))[::-1][:5]
        actual = actual_map.get(uid, np.nan)
        results.append({
            'week_id': forecast_week,
            'item_id': uid,
            'xai_type': 'shap',
            'payload': json.dumps({
                'base_value': round(base, 4),
                'prediction': round(float(preds[i]), 4),
                'actual': round(float(actual), 4) if not np.isnan(actual) else None,
                'error_pct': round(abs(float(preds[i]) - actual) / actual * 100, 2) if actual > 0 else None,
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
    return results


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
    preds_orig = model.predict(X_orig.values).clip(min=0)

    cf_preds: dict[str, np.ndarray] = {}
    for scenario, overrides in _CF_PERTURBATIONS:
        X_cf = X_orig.copy()
        for col, val in overrides.items():
            if col in X_cf.columns:
                X_cf[col] = val
        cf_preds[scenario] = model.predict(X_cf.values).clip(min=0)

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
                        'prediction_cf': round(float(cf_preds[scenario][i]), 4),
                        'delta': round(float(cf_preds[scenario][i]) - orig, 4),
                        'delta_pct': round((float(cf_preds[scenario][i]) - orig) / orig * 100 if orig > 0 else 0, 2),
                    }
                    for scenario, _ in _CF_PERTURBATIONS
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
) -> list[dict]:
    """
    For each bad item find a good reference week (same week-of-year, MAPE < 15%)
    and diff the SHAP profiles. Fetches reference feature rows from SQLite.
    """
    from xai_forecast.db import load_features_week

    bad_woy = pd.Timestamp(forecast_week).isocalendar()[1]
    results = []

    for uid in items:
        good_weeks = all_evals[
            (all_evals['unique_id'] == uid) & (all_evals['mape'] < 15)
        ].copy()
        if good_weeks.empty:
            continue

        same_woy = good_weeks[
            good_weeks['cutoff_week'].apply(
                lambda w: pd.Timestamp(w + ' 00:00:00' if isinstance(w, str) else w).isocalendar()[1] == bad_woy
                if isinstance(w, str) else w.isocalendar()[1] == bad_woy
            )
        ]
        ref_row = same_woy.iloc[-1] if not same_woy.empty else good_weeks.iloc[-1]

        # cutoff_week is the training cutoff; the forecast week is one week later
        ref_cutoff = ref_row['cutoff_week']
        if isinstance(ref_cutoff, str):
            ref_forecast_week = (pd.Timestamp(ref_cutoff) + pd.Timedelta(weeks=1)).strftime('%Y-%m-%d')
        else:
            ref_forecast_week = (ref_cutoff + pd.Timedelta(weeks=1)).strftime('%Y-%m-%d')

        ref_df = load_features_week(conn, ref_forecast_week)
        ref_item = ref_df[ref_df['unique_id'] == uid]
        bad_item = week_df[week_df['unique_id'] == uid]

        if ref_item.empty or bad_item.empty:
            continue

        X_both = pd.concat(
            [bad_item[FEATURE_COLS].fillna(0), ref_item[FEATURE_COLS].fillna(0)],
            ignore_index=True,
        )
        sv = explainer.shap_values(X_both.values)

        diffs = sorted(
            [
                {
                    'feature': FEATURE_COLS[j],
                    'shap_diff': round(float(sv[0][j] - sv[1][j]), 4),
                    'bad_value': round(float(X_both.iloc[0, j]), 4),
                    'good_value': round(float(X_both.iloc[1, j]), 4),
                    'bad_shap': round(float(sv[0][j]), 4),
                    'good_shap': round(float(sv[1][j]), 4),
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
                'good_week': ref_forecast_week,
                'good_week_mape': round(float(ref_row['mape']), 2),
                'top_diffs': diffs[:5],
            }),
        })
    return results
