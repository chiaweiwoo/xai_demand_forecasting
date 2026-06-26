import json
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
    df: pd.DataFrame,
    h1_week: pd.Timestamp,
    items: list[str],
    actual_map: dict[str, float],
) -> list[dict]:
    rows = df[(df['week'] == h1_week) & (df['unique_id'].isin(items))][
        ['unique_id'] + FEATURE_COLS
    ].fillna(0)
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
            'week_id': str(h1_week.date()),
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
    df: pd.DataFrame,
    h1_week: pd.Timestamp,
    items: list[str],
    actual_map: dict[str, float],
) -> list[dict]:
    rows = df[(df['week'] == h1_week) & (df['unique_id'].isin(items))][
        ['unique_id'] + FEATURE_COLS
    ].fillna(0).reset_index(drop=True)
    if rows.empty:
        return []

    X_orig = rows[FEATURE_COLS].copy()
    preds_orig = model.predict(X_orig.values).clip(min=0)

    # Collect per-scenario predictions
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
        scenarios = []
        for scenario, _ in _CF_PERTURBATIONS:
            cf = float(cf_preds[scenario][i])
            scenarios.append({
                'scenario': scenario,
                'prediction_cf': round(cf, 4),
                'delta': round(cf - orig, 4),
                'delta_pct': round((cf - orig) / orig * 100 if orig > 0 else 0, 2),
            })
        results.append({
            'week_id': str(h1_week.date()),
            'item_id': uid,
            'xai_type': 'counterfactual',
            'payload': json.dumps({
                'prediction_original': round(orig, 4),
                'actual': round(float(actual), 4) if not np.isnan(actual) else None,
                'scenarios': scenarios,
            }),
        })
    return results


def contrastive_payloads(
    explainer: shap.TreeExplainer,
    df: pd.DataFrame,
    h1_week: pd.Timestamp,
    items: list[str],
    all_evals: pd.DataFrame,
) -> list[dict]:
    """
    For each bad item, find the most recent good week with similar seasonality
    (same week-of-year, MAPE < 15%) and diff the SHAP profiles.
    """
    bad_woy = h1_week.isocalendar()[1]
    results = []

    for uid in items:
        item_df = df[df['unique_id'] == uid].set_index('week').sort_index()

        good_weeks = all_evals[
            (all_evals['unique_id'] == uid)
            & (all_evals['mape'] < 15)
        ].copy()
        if good_weeks.empty:
            continue

        # Prefer same week-of-year, else fall back to any good week
        same_woy = good_weeks[
            good_weeks['cutoff_week'].apply(
                lambda w: (w + pd.Timedelta(weeks=1)).isocalendar()[1] == bad_woy
            )
        ]
        ref_row = same_woy.iloc[-1] if not same_woy.empty else good_weeks.iloc[-1]
        good_week = ref_row['cutoff_week'] + pd.Timedelta(weeks=1)

        bad_X = item_df.loc[item_df.index == h1_week, FEATURE_COLS]
        good_X = item_df.loc[item_df.index == good_week, FEATURE_COLS]
        if bad_X.empty or good_X.empty:
            continue

        X_both = pd.concat([bad_X, good_X], ignore_index=True).fillna(0)
        sv = explainer.shap_values(X_both[FEATURE_COLS].values)

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
            'week_id': str(h1_week.date()),
            'item_id': uid,
            'xai_type': 'contrastive',
            'payload': json.dumps({
                'bad_week': str(h1_week.date()),
                'good_week': str(good_week.date()),
                'good_week_mape': round(float(ref_row['mape']), 2),
                'top_diffs': diffs[:5],
            }),
        })
    return results
