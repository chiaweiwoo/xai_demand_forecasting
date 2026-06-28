"""
Deterministic evidence detectors. Each fires on real data thresholds and
returns a CandidateFinding with supporting evidence. No LLM calls here.

Detectors:
  over_forecast_bias       -- systematic direction: are all bad weeks over-forecasts?
  dominant_driver          -- one or two SHAP features dominate (>60% of payloads)
  demand_cliff             -- worst items: lag_1 >> actual (demand dropped sharply)
  external_coincidence     -- bad week co-occurs with a notable external condition
  counterfactual_material  -- zeroing SNAP/event/price moves prediction > threshold
  contrastive_gap          -- structural SHAP diff exists vs good reference week
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

import pandas as pd

from .schemas import CandidateFinding

_CF_DELTA_THRESHOLD = 5.0    # % change in prediction to call a CF scenario "material"
_DOMINANT_THRESHOLD = 60.0   # % of SHAP payloads for a feature to be "dominant"
_CLIFF_RATIO        = 3.0    # lag_1 / actual ratio to call it a "demand cliff"
_MIN_CLIFF_ITEMS    = 3      # need at least this many cliff items to fire the detector


def _load_shap_rows(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT week_id, item_id, payload FROM xai_results WHERE xai_type='shap'"
    )
    return [dict(r) for r in cur.fetchall()]


def _load_bad_weeks(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT DISTINCT week_id FROM evaluations WHERE is_bad_week=1 ORDER BY week_id"
    )
    return [r[0] for r in cur.fetchall()]


def detect_over_forecast_bias(conn: sqlite3.Connection) -> CandidateFinding | None:
    """Fire when all (or nearly all) bad-week SHAP payloads are over-forecasts."""
    shap_rows = _load_shap_rows(conn)
    if not shap_rows:
        return None

    total = len(shap_rows)
    over = 0
    under = 0
    for row in shap_rows:
        p = json.loads(row['payload'])
        direction = p.get('direction')
        if direction == 'over':
            over += 1
        elif direction == 'under':
            under += 1

    directional = over + under  # payloads with actual > 0 where direction is defined
    if directional == 0:
        return None

    pct_over = over / directional * 100
    if pct_over < 70:
        return None

    return CandidateFinding(
        finding_id='over_forecast_bias',
        finding_type='over_forecast_bias',
        score=pct_over / 100,
        summary=f'{pct_over:.0f}% of bad-week forecasts are over-forecasts ({over}/{directional} directional payloads)',
        evidence={
            'total_payloads': total,
            'directional_payloads': directional,
            'over_count': over,
            'under_count': under,
            'pct_over': round(pct_over, 1),
            'implication': 'Model systematically over-forecasts during bad weeks. Risk is over-ordering / excess inventory, not stockouts.',
        },
    )


def detect_dominant_driver(conn: sqlite3.Connection) -> CandidateFinding | None:
    """Fire when one or two SHAP features appear in >60% of bad-week payloads."""
    shap_rows = _load_shap_rows(conn)
    if not shap_rows:
        return None

    total = len(shap_rows)
    all_weeks: set = set()
    feature_counts: dict[str, int] = defaultdict(int)
    feature_weeks: dict[str, set] = defaultdict(set)

    for row in shap_rows:
        week_id = row['week_id']
        all_weeks.add(week_id)
        p = json.loads(row['payload'])
        for f in p.get('top_features', []):
            feat = f['feature']
            feature_counts[feat] += 1
            feature_weeks[feat].add(week_id)

    n_weeks = len(all_weeks)
    dominant = [
        {
            'feature': feat,
            'count': cnt,
            'pct_payloads': round(cnt / total * 100, 1),
            'pct_bad_weeks': round(len(feature_weeks[feat]) / n_weeks * 100, 1) if n_weeks else 0,
        }
        for feat, cnt in feature_counts.items()
        if cnt / total * 100 >= _DOMINANT_THRESHOLD
    ]

    if not dominant:
        return None

    dominant.sort(key=lambda x: x['pct_payloads'], reverse=True)
    top = dominant[0]

    return CandidateFinding(
        finding_id='dominant_driver',
        finding_type='dominant_driver',
        score=top['pct_payloads'] / 100,
        summary=(
            f"{top['feature']} appears in {top['pct_payloads']}% of bad-week SHAP payloads "
            f"across {top['pct_bad_weeks']}% of bad weeks"
        ),
        evidence={
            'total_payloads': total,
            'n_bad_weeks': n_weeks,
            'dominant_features': dominant,
            'all_feature_counts': sorted(
                [
                    {
                        'feature': f,
                        'count': c,
                        'pct_payloads': round(c / total * 100, 1),
                        'pct_bad_weeks': round(len(feature_weeks[f]) / n_weeks * 100, 1) if n_weeks else 0,
                    }
                    for f, c in feature_counts.items()
                ],
                key=lambda x: x['count'],
                reverse=True,
            )[:10],
        },
    )


def detect_demand_cliff(conn: sqlite3.Connection) -> CandidateFinding | None:
    """
    Fire when multiple worst-performing items show lag_1 >> actual (demand dropped
    sharply after the model anchored on recent high sales).
    """
    shap_rows = _load_shap_rows(conn)
    if not shap_rows:
        return None

    cliff_items = []
    for row in shap_rows:
        p = json.loads(row['payload'])
        actual = p.get('actual')
        pred   = p.get('prediction')
        if actual is None or actual <= 0:
            continue

        lag1_val = None
        for f in p.get('top_features', []):
            if f['feature'] == 'lag_1':
                lag1_val = f.get('feature_value')
                break

        if lag1_val is None or lag1_val <= 0:
            continue

        if lag1_val / max(actual, 0.1) >= _CLIFF_RATIO:
            cliff_items.append({
                'week_id': row['week_id'],
                'item_id': row['item_id'],
                'lag_1_value': round(lag1_val, 2),
                'actual': round(actual, 2),
                'prediction': round(pred, 2) if pred is not None else None,
                'cliff_ratio': round(lag1_val / actual, 1),
                'signed_error_pct': p.get('signed_error'),
            })

    if len(cliff_items) < _MIN_CLIFF_ITEMS:
        return None

    cliff_items.sort(key=lambda x: x['cliff_ratio'], reverse=True)
    score = min(1.0, len(cliff_items) / 50)

    return CandidateFinding(
        finding_id='demand_cliff',
        finding_type='demand_cliff',
        score=score,
        summary=(
            f'{len(cliff_items)} items show demand-cliff pattern: '
            f'lag_1 >= {_CLIFF_RATIO}x actual sales'
        ),
        evidence={
            'n_cliff_items': len(cliff_items),
            'cliff_ratio_threshold': _CLIFF_RATIO,
            'top_examples': cliff_items[:10],
            'interpretation': (
                'The model forecast based on recent high sales (lag_1) but actual demand '
                'had already dropped sharply. This is the hallmark of momentum over-anchoring.'
            ),
        },
    )


def detect_external_coincidence(conn: sqlite3.Connection) -> CandidateFinding | None:
    """
    Fire when one or more bad weeks coincide with a notable external condition:
    gas spike, heat wave, or sentiment crisis. Stated as correlation only — not cause.
    """
    bad_weeks = _load_bad_weeks(conn)
    if not bad_weeks:
        return None

    n_ext = conn.execute('SELECT COUNT(*) FROM external_signals').fetchone()[0]
    if n_ext == 0:
        return None

    ext_rows = conn.execute(
        f"SELECT week, temp_max, heat_days, precip, gas_price, consumer_sentiment "
        f"FROM external_signals WHERE week IN ({','.join('?' * len(bad_weeks))})",
        bad_weeks,
    ).fetchall()

    notable = []
    for row in ext_rows:
        week, temp_max, heat_days, precip, gas, sent = row
        flags = []
        if heat_days is not None and heat_days >= 4:
            flags.append(f'heat_wave ({heat_days} hot days)')
        if gas is not None and gas >= 4.50:
            flags.append(f'gas_spike (${gas:.2f}/gal)')
        elif gas is not None and gas >= 4.00:
            flags.append(f'high_gas (${gas:.2f}/gal)')
        if sent is not None and sent < 65:
            flags.append(f'sentiment_crisis (index={sent:.1f})')
        if flags:
            notable.append({
                'week': week,
                'conditions': flags,
                'temp_max': temp_max,
                'heat_days': heat_days,
                'gas_price': gas,
                'consumer_sentiment': sent,
            })

    if not notable:
        return None

    return CandidateFinding(
        finding_id='external_coincidence',
        finding_type='external_coincidence',
        score=len(notable) / len(bad_weeks),
        summary=(
            f'{len(notable)} of {len(bad_weeks)} bad weeks coincide with notable external conditions'
        ),
        evidence={
            'n_bad_weeks': len(bad_weeks),
            'n_notable': len(notable),
            'notable_weeks': notable,
            'caveat': (
                'These are CORRELATIONS only. The external signals are in the feature set '
                'but their causal role in any specific forecast error cannot be confirmed '
                'from SHAP data alone — the model may not have weighted them significantly.'
            ),
        },
    )


def detect_counterfactual_material(conn: sqlite3.Connection) -> CandidateFinding | None:
    """
    Fire when zeroing SNAP, events, or price change would have moved predictions
    by more than _CF_DELTA_THRESHOLD% for a substantial fraction of items.
    """
    cur = conn.execute(
        "SELECT week_id, item_id, payload FROM xai_results WHERE xai_type='counterfactual'"
    )
    cf_rows = [dict(r) for r in cur.fetchall()]
    if not cf_rows:
        return None

    scenario_impacts: dict[str, list[float]] = defaultdict(list)
    for row in cf_rows:
        p = json.loads(row['payload'])
        for s in p.get('scenarios', []):
            if s.get('was_active') and abs(s.get('delta_pct') or 0) >= _CF_DELTA_THRESHOLD:
                scenario_impacts[s['scenario']].append(abs(s['delta_pct']))

    if not scenario_impacts:
        return None

    total_items = len(cf_rows)
    summary_rows = [
        {
            'scenario': scenario,
            'n_material_items': len(deltas),
            'pct_items': round(len(deltas) / total_items * 100, 1),
            'avg_delta_pct': round(sum(deltas) / len(deltas), 1),
            'max_delta_pct': round(max(deltas), 1),
        }
        for scenario, deltas in scenario_impacts.items()
    ]
    summary_rows.sort(key=lambda x: x['n_material_items'], reverse=True)

    top = summary_rows[0]
    score = top['pct_items'] / 100

    return CandidateFinding(
        finding_id='counterfactual_material',
        finding_type='counterfactual_material',
        score=score,
        summary=(
            f"Zeroing '{top['scenario']}' moves prediction by >{_CF_DELTA_THRESHOLD}% "
            f"for {top['pct_items']}% of explained items"
        ),
        evidence={
            'total_explained_items': total_items,
            'delta_threshold_pct': _CF_DELTA_THRESHOLD,
            'scenario_impacts': summary_rows,
        },
    )


def detect_contrastive_gap(conn: sqlite3.Connection) -> CandidateFinding | None:
    """
    Summarise contrastive coverage and the most common structural diffs
    between bad weeks and their good-reference counterparts.
    """
    cur = conn.execute(
        "SELECT week_id, item_id, payload FROM xai_results WHERE xai_type='contrastive'"
    )
    cont_rows = [dict(r) for r in cur.fetchall()]

    total_shap = conn.execute(
        "SELECT COUNT(*) FROM xai_results WHERE xai_type='shap'"
    ).fetchone()[0]

    if not cont_rows or not total_shap:
        return None

    coverage_pct = len(cont_rows) / total_shap * 100

    feature_diffs: dict[str, list[float]] = defaultdict(list)
    for row in cont_rows:
        p = json.loads(row['payload'])
        for diff in p.get('top_diffs', [])[:3]:
            feature_diffs[diff['feature']].append(abs(diff['shap_diff']))

    top_diffs = sorted(
        [
            {
                'feature': f,
                'avg_shap_diff': round(sum(vals) / len(vals), 4),
                'n_items': len(vals),
            }
            for f, vals in feature_diffs.items()
        ],
        key=lambda x: x['avg_shap_diff'],
        reverse=True,
    )[:5]

    return CandidateFinding(
        finding_id='contrastive_gap',
        finding_type='contrastive_gap',
        score=coverage_pct / 100,
        summary=(
            f'Contrastive coverage: {coverage_pct:.0f}% ({len(cont_rows)}/{total_shap} items). '
            f'Top structural diff: {top_diffs[0]["feature"] if top_diffs else "n/a"}'
        ),
        evidence={
            'total_shap_items': total_shap,
            'contrastive_items': len(cont_rows),
            'coverage_pct': round(coverage_pct, 1),
            'coverage_gap_note': (
                f'{100 - coverage_pct:.0f}% of explained items have no same-WOY good '
                'reference week — they cannot be compared to a historical success.'
            ),
            'top_structural_diffs': top_diffs,
        },
    )


def run_all_detectors(conn: sqlite3.Connection) -> list[CandidateFinding]:
    """Run all detectors and return non-None results sorted by score descending."""
    detectors = [
        detect_over_forecast_bias,
        detect_dominant_driver,
        detect_demand_cliff,
        detect_external_coincidence,
        detect_counterfactual_material,
        detect_contrastive_gap,
    ]
    findings = []
    for fn in detectors:
        try:
            result = fn(conn)
            if result is not None:
                findings.append(result)
        except Exception as exc:
            print(f'  [WARN] detector {fn.__name__} failed: {exc}')
    findings.sort(key=lambda x: x.score, reverse=True)
    return findings
