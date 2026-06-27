"""
LLM narrative layer — generates human-readable explanations for bad weeks.

Uses DeepSeek V4 Flash (OpenAI-SDK compatible).
If DEEPSEEK_API_KEY is not set or openai is not installed, all generate()
calls return None silently — the dashboard falls back to charts only.

Dossier builders (build_*) are pure functions: no network, fully unit-testable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]
    _OPENAI_AVAILABLE = False

# DeepSeek V4 Flash generates ~800 tokens on verbose week dossiers (observed in production).
# 2000 leaves a comfortable margin. Never lower this value.
MAX_NARRATIVE_TOKENS = 2000


# ── Prompt constants ──────────────────────────────────────────────────────────
# NOTE: These are *_PROMPT constants. Per project rules, run /prompt-audit
# before committing any edit to these strings.

WEEK_NARRATIVE_PROMPT = """\
You are a retail demand forecasting analyst. A business leader is reviewing a week \
where the model failed badly. Write a concise narrative explaining WHY the model \
underperformed this week, grounded ONLY in the evidence provided.

Rules:
- Use ONLY features and numbers from the evidence JSON. Do NOT invent drivers.
- Output valid JSON matching this schema exactly:
  {"headline": "<10-15 word summary>", "body": "<2-3 sentences, plain English, no jargon>", \
"primary_driver": "<exact feature name from the features list>", "confidence": "<high|medium|low>"}
- confidence = high if top driver's pct_of_top_features > 30, medium if 15-30, low otherwise. \
If n_skus_explained < 3, set confidence to low regardless of pct_of_top_features.
- If evidence is sparse (n_skus_explained < 3 or top_features is empty), flag this in body: \
"Limited data this week — interpret with caution."
- Never mention "SHAP", "log-margin", or model internals in body — translate to plain English.
- If lag_1 or lag_2 dominates: say "recent sales trend unexpectedly changed". \
If snap dominates: say "promotion-week uplift". If price_change_pct: say "pricing change". \
If lag_52 dominates: say "year-over-year seasonality mismatch".
- Respond in English only.
- Before writing your final JSON, verify: (1) primary_driver is a name from the features list, \
(2) body contains no model jargon, (3) all numbers came from the evidence JSON."""

ITEM_NARRATIVE_PROMPT = """\
You are a retail demand forecasting analyst explaining a specific product's forecast error.
Given the evidence JSON for one SKU, write a concise narrative grounded ONLY in the data provided.

Rules:
- Use ONLY features and numbers from the evidence JSON. Do NOT invent drivers.
- Output valid JSON matching this schema exactly:
  {"headline": "<10-15 word summary>", "body": "<2-3 sentences, plain English, no jargon>", \
"primary_driver": "<exact feature name from the features list>", "confidence": "<high|medium|low>"}
- confidence = high if top SHAP feature has |value| > 0.5, medium 0.2-0.5, low if < 0.2.
- If contrastive data is present, mention what was structurally different vs the good reference week. \
If contrastive data is absent, do not mention comparisons to past weeks.
- Never mention "SHAP", "log-margin", or model internals in body — translate to plain English.
- direction field: "over" means model over-forecast (predicted more than actual), \
"under" means under-forecast.
- Respond in English only.
- Before writing your final JSON, verify: (1) primary_driver is a name from the features list, \
(2) body contains no model jargon, (3) no past-week comparisons unless contrastive data is present."""

EXECUTIVE_NARRATIVE_PROMPT = """\
You are a retail demand forecasting analyst presenting findings to a business leader.
Given a summary of which features most often drove forecast failures across multiple bad weeks, \
write an executive synthesis.

Rules:
- Use ONLY features and patterns from the evidence JSON. Do NOT invent observations.
- Output valid JSON matching this schema exactly:
  {"headline": "<12-18 word executive summary>", \
"body": "<3-4 sentences, plain English suitable for a business meeting>", \
"primary_driver": "<most recurring feature name from the features list>", "confidence": "<high|medium|low>"}
- confidence = high if top feature's pct_bad_weeks > 60, medium if 40-60, low otherwise. \
Use the pct_bad_weeks field from top_recurring_features — not pct_payloads. \
If n_bad_weeks < 5, set confidence to low regardless of feature frequency.
- If n_bad_weeks < 5, note in body that the pattern is based on limited data.
- Focus on what this means for the business, not for the model's internals.
- Respond in English only.
- Before writing your final JSON, verify: (1) primary_driver is a name from the features list, \
(2) all percentages came from the evidence JSON, (3) body is suitable for a business meeting audience."""


# ── Dossier builders (pure — no network) ─────────────────────────────────────

def build_week_dossier(
    forecast_week: str,
    shap_rows: list[dict],
    wmape_zscore: float | None,
    n_items_in_week: int,
) -> dict[str, Any]:
    """
    Build evidence dossier for week-level narrative.
    shap_rows: list of xai_results rows (each has a 'payload' JSON string).
    """
    feature_shap: dict[str, list[float]] = {}
    for row in shap_rows:
        p = json.loads(row['payload']) if isinstance(row.get('payload'), str) else row.get('payload', {})
        for f in p.get('top_features', []):
            feature_shap.setdefault(f['feature'], []).append(abs(f['shap_value']))

    top_features: list[dict] = sorted(
        [
            {
                'feature': feat,
                'mean_abs_shap': round(float(sum(vs) / len(vs)), 4),
                'n_skus': len(vs),
            }
            for feat, vs in feature_shap.items()
        ],
        key=lambda x: x['mean_abs_shap'],
        reverse=True,
    )[:7]

    total_shap = sum(f['mean_abs_shap'] for f in top_features)
    for f in top_features:
        f['pct_of_top_features'] = round(f['mean_abs_shap'] / total_shap * 100, 1) if total_shap > 0 else 0.0

    return {
        'forecast_week': forecast_week,
        'wmape_zscore': round(float(wmape_zscore), 2) if wmape_zscore is not None else None,
        'n_items_in_week': n_items_in_week,
        'n_skus_explained': len(shap_rows),
        'top_features': top_features,
        'features': [f['feature'] for f in top_features],
    }


def build_item_dossier(
    forecast_week: str,
    item_id: str,
    shap_payload: dict | None,
    cf_payload: dict | None,
    cont_payload: dict | None,
) -> dict[str, Any]:
    """
    Build evidence dossier for item-level narrative.
    Each payload is a parsed JSON dict (not a string).
    """
    dossier: dict[str, Any] = {
        'forecast_week': forecast_week,
        'item_id': item_id,
        'features': [],
    }

    if shap_payload:
        dossier['prediction'] = shap_payload.get('prediction')
        dossier['actual'] = shap_payload.get('actual')
        dossier['error_pct'] = shap_payload.get('error_pct')
        dossier['direction'] = shap_payload.get('direction')
        top5 = shap_payload.get('top_features', [])
        dossier['top_features'] = top5[:5]
        dossier['features'] = [f['feature'] for f in top5]

    if cf_payload:
        active_cf = [
            {'scenario': s['scenario'], 'delta_pct': s.get('delta_pct')}
            for s in cf_payload.get('scenarios', [])
            if s.get('was_active') and abs(s.get('delta_pct') or 0) > 2
        ]
        if active_cf:
            dossier['active_counterfactuals'] = active_cf

    if cont_payload:
        dossier['contrastive'] = {
            'good_week': cont_payload.get('good_week'),
            'good_week_mape': cont_payload.get('good_week_mape'),
            'top_diffs': cont_payload.get('top_diffs', [])[:3],
        }

    return dossier


def compute_recurring_drivers(shap_rows: list[dict]) -> list[dict]:
    """
    Count feature appearances across all SHAP payloads.
    shap_rows: list of xai_results rows (each has 'week_id' and a 'payload' JSON string or dict).
    Returns list[{feature, count, pct_payloads, n_weeks, pct_bad_weeks}] sorted by count desc.
    Single source of truth — used by both backtest.py and app.py.
    """
    from collections import defaultdict
    feature_counts: dict[str, int] = defaultdict(int)
    feature_weeks: dict[str, set] = defaultdict(set)
    all_weeks: set = set()
    total = len(shap_rows)
    for row in shap_rows:
        week_id = row.get('week_id', '')
        if week_id:
            all_weeks.add(week_id)
        p = json.loads(row['payload']) if isinstance(row.get('payload'), str) else row.get('payload', {})
        for f in p.get('top_features', []):
            feature_counts[f['feature']] += 1
            if week_id:
                feature_weeks[f['feature']].add(week_id)
    n_distinct_weeks = len(all_weeks)
    return sorted(
        [
            {
                'feature': feat,
                'count': cnt,
                'pct_payloads': round(cnt / total * 100, 1) if total else 0,
                'n_weeks': len(feature_weeks[feat]),
                'pct_bad_weeks': round(len(feature_weeks[feat]) / n_distinct_weeks * 100, 1)
                    if n_distinct_weeks else 0,
            }
            for feat, cnt in feature_counts.items()
        ],
        key=lambda x: x['count'],
        reverse=True,
    )


def build_executive_dossier(
    drivers_list: list[dict],
    n_bad_weeks: int,
    n_total_weeks: int,
) -> dict[str, Any]:
    """
    Build evidence dossier for executive synthesis.
    drivers_list: [{feature, count, pct_payloads}, ...] sorted by count desc.
    """
    return {
        'n_bad_weeks': n_bad_weeks,
        'n_total_weeks': n_total_weeks,
        'bad_week_rate_pct': round(n_bad_weeks / n_total_weeks * 100, 1) if n_total_weeks > 0 else 0,
        'top_recurring_features': drivers_list[:7],
        'features': [d['feature'] for d in drivers_list[:7]],
    }


# ── Grounding check ────────────────────────────────────────────────────────────

def _grounding_check(narrative: dict, dossier: dict) -> bool:
    """
    Structural check: primary_driver must be a feature name present in the evidence.
    Does NOT validate free-text body content or numbers — low temperature + prompt
    rules are the main mitigation for those.
    Returns False (flags for confidence downgrade) if primary_driver is absent or unknown.
    """
    primary = narrative.get('primary_driver', '')
    if not primary:
        return False
    evidence_features = dossier.get('features', [])
    if not evidence_features:
        return True  # can't verify
    return primary in evidence_features


# ── Narrator ──────────────────────────────────────────────────────────────────

class DeepSeekNarrator:
    """
    Thin wrapper over the OpenAI-compatible DeepSeek API.

    Instantiate once; call generate() per narrative.
    Returns None if the key is absent, openai is not installed, or the call fails.
    """

    def __init__(self) -> None:
        self._key = os.environ.get('DEEPSEEK_API_KEY')
        self.model_id = os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash')
        self._base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        self._client = None
        if self._key and _OPENAI_AVAILABLE and _OpenAI is not None:
            self._client = _OpenAI(api_key=self._key, base_url=self._base_url)

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, system_prompt: str, dossier: dict) -> dict | None:
        """
        Call the LLM with the given system prompt and dossier as user message.
        Returns the parsed narrative dict or None on any failure.
        """
        if not self.available:
            return None
        try:
            resp = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model_id,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': json.dumps(dossier, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=MAX_NARRATIVE_TOKENS,
                response_format={'type': 'json_object'},
                timeout=30,
            )
            choice = resp.choices[0]
            if choice.finish_reason == 'length':
                logger.warning(
                    'Narrative truncated at max_tokens=%d (finish_reason=length). '
                    'Increase MAX_NARRATIVE_TOKENS in narrate.py.',
                    MAX_NARRATIVE_TOKENS,
                )
                return None
            raw = choice.message.content
            result: dict = json.loads(raw)

            for required_key in ('headline', 'body', 'primary_driver', 'confidence'):
                if required_key not in result:
                    logger.warning('Narrative missing required key: %s', required_key)
                    return None

            if not _grounding_check(result, dossier):
                result['confidence'] = 'low'
                result['grounding_warning'] = True

            result['model'] = self.model_id
            return result

        except Exception as exc:
            logger.warning('Narrative generation failed: %s', exc)
            return None
