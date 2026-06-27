"""
Read-only tools the LangGraph agent can call to pull context on demand.

Each function is a clean DB read — no LLM calls, no side-effects.
The agent decides which tools to call per finding type.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd


def read_forecast_accuracy(conn: sqlite3.Connection) -> dict[str, Any]:
    """Weekly MAPE series + bad/good week flags across the full backtest."""
    df = pd.read_sql(
        """SELECT week_id,
                  AVG(h1_mape)     AS avg_mape,
                  AVG(mape_zscore) AS avg_zscore,
                  MAX(is_bad_week) AS is_bad_week,
                  COUNT(*)         AS n_items
           FROM evaluations
           GROUP BY week_id
           ORDER BY week_id""",
        conn,
    )
    bad = df[df['is_bad_week'] == 1]
    good = df[df['is_bad_week'] == 0]
    return {
        'total_weeks': len(df),
        'n_bad_weeks': int(bad['is_bad_week'].sum()),
        'n_good_weeks': len(good),
        'bad_week_rate_pct': round(len(bad) / len(df) * 100, 1) if len(df) else 0,
        'overall_avg_mape': round(float(df['avg_mape'].mean()), 2),
        'bad_weeks_avg_mape': round(float(bad['avg_mape'].mean()), 2) if len(bad) else None,
        'worst_week': df.loc[df['avg_mape'].idxmax(), 'week_id'] if len(df) else None,
        'worst_week_mape': round(float(df['avg_mape'].max()), 2) if len(df) else None,
        'bad_weeks': bad[['week_id', 'avg_mape', 'avg_zscore']].round(2).to_dict('records'),
    }


def read_bad_weeks(conn: sqlite3.Connection) -> list[dict]:
    """All flagged bad weeks with their z-scores."""
    rows = conn.execute(
        """SELECT DISTINCT week_id,
                  AVG(mape_zscore) AS avg_zscore,
                  COUNT(*) AS n_items,
                  AVG(h1_mape) AS avg_mape
           FROM evaluations WHERE is_bad_week=1
           GROUP BY week_id ORDER BY week_id"""
    ).fetchall()
    return [
        {
            'week_id': r[0],
            'avg_zscore': round(r[1], 2) if r[1] else None,
            'n_items': r[2],
            'avg_mape': round(r[3], 2) if r[3] else None,
        }
        for r in rows
    ]


def read_good_weeks_for_item(conn: sqlite3.Connection, item_id: str) -> list[dict]:
    """Good reference weeks (MAPE < 15%) for a specific SKU."""
    rows = conn.execute(
        "SELECT week_id, h1_mape FROM evaluations WHERE item_id=? AND h1_mape < 15 ORDER BY week_id",
        (item_id,),
    ).fetchall()
    return [{'week_id': r[0], 'mape': round(r[1], 2)} for r in rows]


def read_xai_findings(
    conn: sqlite3.Connection,
    week_id: str | None = None,
    item_id: str | None = None,
    xai_type: str | None = None,
) -> list[dict]:
    """XAI payloads — can filter by week, item, and/or type."""
    clauses = []
    params: list = []
    if week_id:
        clauses.append('week_id = ?')
        params.append(week_id)
    if item_id:
        clauses.append('item_id = ?')
        params.append(item_id)
    if xai_type:
        clauses.append('xai_type = ?')
        params.append(xai_type)

    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    rows = conn.execute(
        f'SELECT week_id, item_id, xai_type, payload FROM xai_results {where} LIMIT 200',
        params,
    ).fetchall()
    return [
        {
            'week_id': r[0],
            'item_id': r[1],
            'xai_type': r[2],
            'payload': json.loads(r[3]),
        }
        for r in rows
    ]


def read_demand_trajectory(
    conn: sqlite3.Connection,
    item_id: str,
    around_week: str,
    n_weeks_before: int = 8,
    n_weeks_after: int = 2,
) -> dict[str, Any]:
    """
    Internal time series for one SKU around a bad week:
    actual sales, features (lag_1), and forecast vs actual.
    """
    sales = pd.read_sql(
        """SELECT ws.week, ws.y AS actual_sales, f.lag_1, f.rolling_4_mean,
                  fc.h1 AS forecast
           FROM weekly_sales ws
           LEFT JOIN features   f  ON f.week = ws.week AND f.unique_id = ws.unique_id
           LEFT JOIN forecasts  fc ON fc.week_id = ws.week AND fc.item_id = ws.unique_id
           WHERE ws.unique_id = ? AND ws.week <= ?
           ORDER BY ws.week DESC
           LIMIT ?""",
        conn,
        params=(item_id, around_week, n_weeks_before + n_weeks_after + 1),
    )
    sales = sales.sort_values('week').tail(n_weeks_before + n_weeks_after + 1)
    return {
        'item_id': item_id,
        'around_week': around_week,
        'trajectory': sales.round(2).to_dict('records'),
    }


def read_external_signals(conn: sqlite3.Connection, week_id: str) -> dict[str, Any] | None:
    """External signals for one specific week."""
    row = conn.execute(
        'SELECT * FROM external_signals WHERE week = ?', (week_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    # Add human-readable labels
    heat = d.get('heat_days') or 0
    gas  = d.get('gas_price') or 0
    sent = d.get('consumer_sentiment') or 0
    d['weather_label']    = 'heat_wave' if heat >= 4 else ('hot' if heat >= 2 else 'normal')
    d['gas_label']        = 'historic_spike' if gas >= 4.50 else ('high' if gas >= 4.00 else ('low' if gas <= 2.60 else 'normal'))
    d['sentiment_label']  = 'crisis_low' if sent < 65 else ('below_average' if sent < 80 else ('high_confidence' if sent >= 90 else 'average'))
    return d


def read_model_metadata(conn: sqlite3.Connection) -> dict[str, Any]:
    """
    Model training configuration and global feature importance from checkpoints.
    Reads week_to_cutoff.json; loads one checkpoint for global importance.
    """
    import os
    import json as _json

    meta: dict[str, Any] = {
        'training_window_weeks': 156,
        'retrain_freq_weeks': 4,
        'objective': 'tweedie (variance_power=1.5)',
        'shap_space': 'log-margin (Tweedie log-link)',
        'store': 'CA_1 only (~3049 SKUs)',
    }

    cutoff_path = 'db/week_to_cutoff.json'
    if os.path.exists(cutoff_path):
        with open(cutoff_path) as f:
            week_cutoff = _json.load(f)
        meta['n_week_cutoff_mappings'] = len(week_cutoff)
        cutoffs = sorted(set(week_cutoff.values()))
        meta['n_checkpoints'] = len(cutoffs)

        # Try loading the last checkpoint for global feature importance
        last_cutoff = cutoffs[-1] if cutoffs else None
        if last_cutoff:
            ckpt_path = f'models/checkpoint_{last_cutoff}.lgbm'
            if os.path.exists(ckpt_path):
                try:
                    import lightgbm as lgb
                    from xai_forecast.features import FEATURE_COLS
                    model = lgb.Booster(model_file=ckpt_path)
                    importances = model.feature_importance(importance_type='gain')
                    total_imp = float(importances.sum()) or 1.0
                    fi = sorted(
                        [
                            {
                                'feature': FEATURE_COLS[i],
                                'importance_gain': round(float(importances[i]), 2),
                                'pct': round(float(importances[i]) / total_imp * 100, 1),
                            }
                            for i in range(len(FEATURE_COLS))
                        ],
                        key=lambda x: x['importance_gain'],
                        reverse=True,
                    )
                    meta['global_feature_importance'] = fi[:10]
                    meta['checkpoint_used'] = ckpt_path
                except Exception as exc:
                    meta['feature_importance_error'] = str(exc)
    return meta


def read_recurring_drivers(conn: sqlite3.Connection) -> list[dict]:
    """Feature appearance frequency across all bad-week SHAP payloads."""
    from collections import defaultdict

    rows = conn.execute(
        "SELECT week_id, payload FROM xai_results WHERE xai_type='shap'"
    ).fetchall()

    total = len(rows)
    if not total:
        return []

    feature_counts: dict[str, int] = defaultdict(int)
    feature_weeks: dict[str, set] = defaultdict(set)
    all_weeks: set = set()

    for row in rows:
        week_id = row[0]
        all_weeks.add(week_id)
        p = json.loads(row[1])
        for f in p.get('top_features', []):
            feat = f['feature']
            feature_counts[feat] += 1
            feature_weeks[feat].add(week_id)

    n_weeks = len(all_weeks)
    return sorted(
        [
            {
                'feature': feat,
                'count': cnt,
                'pct_payloads': round(cnt / total * 100, 1),
                'pct_bad_weeks': round(len(feature_weeks[feat]) / n_weeks * 100, 1) if n_weeks else 0,
            }
            for feat, cnt in feature_counts.items()
        ],
        key=lambda x: x['count'],
        reverse=True,
    )
