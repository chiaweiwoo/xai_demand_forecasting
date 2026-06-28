"""
XAI Demand Forecasting — management storytelling dashboard.

Reads as a briefing top-to-bottom:
  1. Verdict (hero)          — the one thing to know, bound live to the analysis
  2. What am I looking at?    — dataset + the question + the 4-step approach
  3. How we flag a bad week  — plain-language definition + MAPE chart
  4. What we found           — accepted findings as story cards
  5. What to do              — decision-ready recommendations (business + DS)
  6. Technical evidence       — findings ledger + per-item XAI (toggle, off by default)

Usage:
    uv run streamlit run app.py
"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from xai_forecast.db import (
    get_conn, week_summary, load_evaluations, load_xai,
    load_insight_summary, load_insight_findings,
)
from xai_forecast.features import FEATURE_COLS

DB_PATH = 'db/forecasting.db'

st.set_page_config(page_title='Demand Forecast — Model Review', layout='centered')

# ── Minimal design system ───────────────────────────────────────────────────────

ACCENT_OVER  = '#c0392b'   # over-stock risk (red)
ACCENT_UNDER = '#1f78b4'   # under-stock risk (blue)
ACCENT_MIXED = '#7f8c8d'   # mixed (grey)
INK          = '#1a1a2e'

st.markdown(
    """
    <style>
      .hero {
        padding: 8px 0 4px;
      }
      .hero h1 {
        font-size: 2.5rem; line-height: 1.15; font-weight: 800;
        margin: 0 0 12px; letter-spacing: -0.02em;
      }
      .hero p.sub {
        font-size: 1.1rem; color: #444; margin: 0 0 18px; max-width: 40rem;
      }
      .risk-badge {
        display: inline-block; padding: 6px 14px; border-radius: 999px;
        font-size: 0.85rem; font-weight: 700; letter-spacing: 0.04em;
        color: #fff; text-transform: uppercase;
      }
      .tile-num { font-size: 2.0rem; font-weight: 800; line-height: 1; }
      .tile-lbl { font-size: 0.85rem; color: #555; margin-top: 6px; }
      .step {
        font-size: 0.9rem; color: #333;
      }
      .step b { color: #1a1a2e; }
      .section-kicker {
        font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #888; margin-bottom: 2px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loaders (ttl so a fresh generate_insights.py shows up, not a stale cache) ─

@st.cache_data(ttl=300)
def _week_summary():
    with get_conn(DB_PATH) as conn:
        return week_summary(conn)


@st.cache_data(ttl=300)
def _evaluations():
    with get_conn(DB_PATH) as conn:
        return load_evaluations(conn)


@st.cache_data(ttl=300)
def _insight_summary():
    with get_conn(DB_PATH) as conn:
        return load_insight_summary(conn)


@st.cache_data(ttl=300)
def _insight_findings():
    with get_conn(DB_PATH) as conn:
        return load_insight_findings(conn)


@st.cache_data(ttl=300)
def _direction_stats() -> dict:
    """Count over/under forecasts across all bad-week SHAP payloads."""
    import json as _json
    with get_conn(DB_PATH) as conn:
        rows = conn.execute(
            """SELECT xr.payload
               FROM xai_results xr
               JOIN evaluations e ON e.week_id = xr.week_id AND e.item_id = xr.item_id
               WHERE xr.xai_type = 'shap' AND e.is_bad_week = 1"""
        ).fetchall()
    over = under = 0
    for (payload,) in rows:
        d = _json.loads(payload).get('direction')
        if d == 'over':
            over += 1
        elif d == 'under':
            under += 1
    total = over + under
    return {'over': over, 'under': under, 'total': total,
            'pct_over': round(over / total * 100, 1) if total else None}


@st.cache_data(ttl=300)
def _actual_vs_forecast():
    with get_conn(DB_PATH) as conn:
        return pd.read_sql(
            '''SELECT f.week_id,
                      SUM(f.h1)  AS total_forecast,
                      SUM(ws.y)  AS total_actual
               FROM forecasts f
               JOIN weekly_sales ws ON ws.week = f.week_id AND ws.unique_id = f.item_id
               GROUP BY f.week_id
               ORDER BY f.week_id''',
            conn,
        )


def _xai(week_id: str, item_id: str) -> dict[str, dict]:
    with get_conn(DB_PATH) as conn:
        rows = load_xai(conn, week_id, item_id)
    return {r['xai_type']: json.loads(r['payload']) for r in rows}


# ── Helpers ─────────────────────────────────────────────────────────────────────

_CONF_BADGE = {'high': ':green[High]', 'medium': ':orange[Medium]', 'low': ':red[Low]'}
_STATUS_BADGE = {
    'accepted':     ':green[Accepted]',
    'rejected':     ':red[Rejected]',
    'needs_review': ':orange[Needs review]',
}

# finding_type → plain, management-friendly card title
_FINDING_TITLE = {
    'over_forecast_bias':       'It only fails in one direction',
    'dominant_driver':          'It over-anchors on recent sales',
    'demand_cliff':             'It misses demand cliffs',
    'counterfactual_material':  'Food-stamp (SNAP) weeks swing the forecast',
    'contrastive_gap':          'Most products lack a fair comparison',
    'external_coincidence':     'Weather / fuel prices — tested, not proven',
}

# raw feature name → plain English (for the hero "root cause" tile)
_FEATURE_PLAIN = {
    'rolling_4_mean':  'the last 4 weeks of sales',
    'rolling_8_mean':  'the last 8 weeks of sales',
    'rolling_13_mean': 'the last 13 weeks of sales',
    'lag_1':           "last week's sales",
    'lag_2':           'sales two weeks ago',
    'lag_52':          'the same week last year',
    'temp_mean':       'average weekly temperature',
    'temp_max':        'peak weekly temperature',
    'temp_min':        'lowest weekly temperature',
    'precip':          'weekly rainfall',
    'heat_days':       'number of hot days that week',
    'gas_price':       'California gas price',
    'consumer_sentiment': 'consumer confidence index',
}

_RISK = {
    'over-stock':  (ACCENT_OVER,  'Over-stock risk'),
    'under-stock': (ACCENT_UNDER, 'Under-stock risk'),
    'mixed':       (ACCENT_MIXED, 'Mixed risk'),
}


def _confidence_badge(conf: str) -> str:
    return _CONF_BADGE.get(conf, conf)


def _status_badge(status: str) -> str:
    return _STATUS_BADGE.get(status, status)


def _accepted(findings: list[dict], ftype: str) -> dict | None:
    return next(
        (f for f in findings if f['finding_type'] == ftype and f['status'] == 'accepted'),
        None,
    )


def _biz_text(finding: dict) -> str:
    """Pull the business-facing explanation out of a finding's hypothesis JSON."""
    hyp = finding.get('hypothesis') or {}
    exp = hyp.get('explanation', {})
    if isinstance(exp, str):
        try:
            exp = json.loads(exp)
        except Exception:
            exp = {}
    return exp.get('business_explanation') or hyp.get('headline') or ''


# ══════════════════════════════════════════════════════════════════════════════
# Load everything
# ══════════════════════════════════════════════════════════════════════════════

try:
    summary = _week_summary()
except Exception:
    st.error('No data yet. Run the pipeline: `backtest.py → run_xai.py → generate_insights.py`')
    st.stop()

if summary.empty:
    st.warning('No evaluations found. Run `backtest.py` first.')
    st.stop()

insights = _insight_summary()
findings = _insight_findings()

n_bad_weeks = int((summary['n_bad_items'] > 0).sum())
n_total_weeks = len(summary)


# ══════════════════════════════════════════════════════════════════════════════
# 1. HERO — the verdict
# ══════════════════════════════════════════════════════════════════════════════

if not insights:
    st.title('Demand Forecast — Model Review')
    st.info(
        'The analysis has not been generated yet. Run:\n\n'
        '`uv run python generate_insights.py`'
    )
    st.stop()

biz = insights['business_leader']
ds  = insights['data_scientist']
overall_conf = (ds.get('overall_confidence')
                or biz.get('overall_confidence')
                or 'medium')

risk_dir = (biz.get('risk_direction') or 'mixed').lower()
risk_color, risk_label = _RISK.get(risk_dir, (ACCENT_MIXED, 'Risk'))

verdict = biz.get('headline') or 'The forecasting model shows a recurring failure pattern.'

st.markdown(
    f"""
    <div class="hero">
      <div class="section-kicker">Model review · Walmart demand forecast</div>
      <h1>{verdict}</h1>
      <span class="risk-badge" style="background:{risk_color}">{risk_label} · Confidence {overall_conf}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write('')

# Three "so what" tiles — bound live to the data
over_f   = _accepted(findings, 'over_forecast_bias')
driver_f = _accepted(findings, 'dominant_driver')

pct_over = None
if over_f:
    pct_over = (over_f.get('evidence') or {}).get('pct_over')

# Fallback: compute direction directly from DB if the detector didn't fire
_dir_stats = _direction_stats()
if pct_over is None and _dir_stats.get('pct_over') is not None:
    pct_over = _dir_stats['pct_over']
_dir_lopsided = pct_over is not None and (pct_over >= 70 or pct_over <= 30)

root_cause = None
if driver_f:
    dom = (driver_f.get('evidence') or {}).get('dominant_features') or []
    if dom:
        feat = dom[0].get('feature', '')
        root_cause = _FEATURE_PLAIN.get(feat, feat)

t1, t2, t3 = st.columns(3)
with t1:
    st.markdown(
        f'<div class="tile-num" style="color:{INK}">{n_bad_weeks}</div>'
        f'<div class="tile-lbl">weeks flagged as problems<br>(out of {n_total_weeks} reviewed)</div>',
        unsafe_allow_html=True,
    )
with t2:
    if pct_over is not None:
        _tile_color = ACCENT_OVER if _dir_lopsided else INK
        _under_pct = round(100 - pct_over, 1)
        _lbl = (
            'of bad-week forecasts were<br>over-forecasts'
            if _dir_lopsided
            else f'{_under_pct:.0f}% under · direction is mixed'
        )
        st.markdown(
            f'<div class="tile-num" style="color:{_tile_color}">{pct_over:.0f}%</div>'
            f'<div class="tile-lbl">over-forecast<br>{_lbl}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="tile-num" style="color:{INK}">—</div>'
            f'<div class="tile-lbl">direction of error</div>',
            unsafe_allow_html=True,
        )
with t3:
    if root_cause:
        st.markdown(
            f'<div class="tile-num" style="color:{INK};font-size:1.3rem;line-height:1.2">{root_cause}</div>'
            f'<div class="tile-lbl">the main thing the model<br>over-trusts when it fails</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="tile-num" style="color:{INK}">—</div>'
            f'<div class="tile-lbl">main driver</div>',
            unsafe_allow_html=True,
        )

st.caption('Bottom line and recommended actions are below ↓')


# ══════════════════════════════════════════════════════════════════════════════
# 2. WHAT AM I LOOKING AT?
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<div class="section-kicker">What am I looking at?</div>', unsafe_allow_html=True)
st.markdown('#### A health-check on the demand forecasting model')

st.markdown(
    '**The data.** Five years of real Walmart sales (2011–2016), one California store, '
    'about 3,000 products. A standard public benchmark dataset (M5).'
)
st.markdown(
    '**The question.** When the model forecasts badly in a given week, *why* does it happen — '
    'and is it the same reason each time?'
)

s1, s2, s3, s4 = st.columns(4)
s1.markdown('<div class="step"><b>1. Forecast</b><br>Predict next-week sales for every product.</div>', unsafe_allow_html=True)
s2.markdown('<div class="step"><b>2. Flag</b><br>Find the weeks where it went genuinely wrong.</div>', unsafe_allow_html=True)
s3.markdown('<div class="step"><b>3. Explain</b><br>Open up each bad week and trace what drove the error.</div>', unsafe_allow_html=True)
s4.markdown('<div class="step"><b>4. Verify</b><br>An AI critic confirms or rejects every explanation.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. HOW WE FLAG A BAD WEEK
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<div class="section-kicker">How we decide a week is "bad"</div>', unsafe_allow_html=True)
st.markdown('#### Not just high error — an *unusual spike* in error')

st.markdown(
    'A product that is always 30% off isn\'t news. A week where the error *spikes far above '
    'its own recent track record* **is**. We flag a week only when the forecast diverges '
    'unusually from actual sales — so we catch genuine surprises, not chronic noise.'
)

avf = _actual_vs_forecast()
avf['week_dt'] = pd.to_datetime(avf['week_id'])

summary_plot = summary.copy()
summary_plot['week_dt'] = pd.to_datetime(summary_plot['week_id'])
bad_weeks_set = set(summary_plot[summary_plot['n_bad_items'] > 0]['week_id'])

fig = go.Figure()

# Shade bad weeks as vertical rectangles
for bw in sorted(bad_weeks_set):
    bw_dt = pd.to_datetime(bw)
    fig.add_vrect(
        x0=bw_dt - pd.Timedelta(days=3),
        x1=bw_dt + pd.Timedelta(days=3),
        fillcolor=ACCENT_OVER, opacity=0.12,
        line_width=0,
    )

fig.add_trace(go.Scatter(
    x=avf['week_dt'], y=avf['total_actual'],
    mode='lines', name='Actual sales',
    line=dict(color='#2c7bb6', width=2),
))
fig.add_trace(go.Scatter(
    x=avf['week_dt'], y=avf['total_forecast'],
    mode='lines', name='Forecast',
    line=dict(color='#888', width=1.6, dash='dot'),
))
fig.update_layout(
    xaxis_title=None,
    yaxis_title='Total weekly units (store-wide)',
    hovermode='x unified',
    height=320,
    margin=dict(t=10, b=10, l=10, r=10),
    legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0),
)
st.plotly_chart(fig, width='stretch')
st.caption(
    f'Blue = actual sales, grey dashed = model forecast. '
    f'Red bands = the {n_bad_weeks} flagged bad weeks (out of {n_total_weeks} reviewed). '
    'A week is flagged when the forecast-vs-actual gap spikes unusually against the model\'s own recent track record.'
)


# ══════════════════════════════════════════════════════════════════════════════
# 4. WHAT WE FOUND
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<div class="section-kicker">What we found</div>', unsafe_allow_html=True)
st.markdown('#### The confirmed patterns behind the bad weeks')

# Order strongest-first; only accepted findings appear as cards.
_CARD_ORDER = [
    'over_forecast_bias',
    'dominant_driver',
    'demand_cliff',
    'counterfactual_material',
    'contrastive_gap',
]
accepted = [f for f in findings if f['status'] == 'accepted']
ordered = [f for ft in _CARD_ORDER for f in accepted if f['finding_type'] == ft]
# any accepted finding not in the explicit order, appended at the end
ordered += [f for f in accepted if f['finding_type'] not in _CARD_ORDER]

if not ordered:
    st.info('No findings cleared the quality gate. Re-run `generate_insights.py`.')
else:
    for f in ordered:
        title = _FINDING_TITLE.get(f['finding_type'], f['finding_type'])
        with st.container(border=True):
            st.markdown(f"**{title}**  ·  {_confidence_badge(f['confidence'])}")
            body = _biz_text(f)
            if body:
                st.markdown(body)

# Honest footnote: rejected findings = rigor signal
rejected = [f for f in findings if f['status'] == 'rejected']
if rejected:
    rej_names = ', '.join(_FINDING_TITLE.get(f['finding_type'], f['finding_type']).lower() for f in rejected)
    st.caption(
        f'We also tested other explanations ({rej_names}) — the evidence didn\'t hold up, '
        'so our critic rejected them rather than overclaim.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. WHAT TO DO
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<div class="section-kicker">What to do</div>', unsafe_allow_html=True)
st.markdown('#### Recommended next steps')

_BUCKET_LABEL = {
    'feature_engineering': 'Feature engineering',
    'model_param':         'Model parameters',
    'workflow':            'Training workflow',
    'algorithm':           'Algorithm',
}
_EFFORT_LABEL = {'low': 'Low effort', 'medium': 'Medium effort', 'high': 'High effort'}

col_biz, col_ds = st.columns(2)

with col_biz:
    with st.container(border=True):
        st.markdown('##### For the business')

        # Progress block: health verdict + diagnosis
        progress = biz.get('progress', {})
        if progress.get('health_verdict'):
            st.markdown(f"**Status:** {progress['health_verdict']}")
        if progress.get('what_we_diagnosed'):
            st.markdown(progress['what_we_diagnosed'])

        # Phased plan
        plan = biz.get('plan', {})
        phases = plan.get('phases', [])
        if phases:
            st.markdown('**Improvement plan:**')
            for phase in phases:
                name   = phase.get('name', '')
                action = phase.get('action', '')
                risk   = phase.get('risk_if_skipped', '')
                st.markdown(f'- **{name}:** {action}')
                if risk:
                    st.caption(f'Risk if skipped: {risk}')
        if plan.get('expected_impact'):
            st.caption(f"Expected impact: {plan['expected_impact']}")

        lims = biz.get('limitations', [])
        if lims:
            st.markdown('**Known limitations:**')
            for lim in lims:
                st.markdown(f'- {lim}')

        # Fallback for old format (summary / improvement_plan fields)
        if not progress and not phases:
            if biz.get('summary'):
                st.markdown(biz['summary'])
            if biz.get('improvement_plan'):
                st.markdown(f"**Plan:** {biz['improvement_plan']}")
            for lim in biz.get('limitations', []):
                st.markdown(f'- {lim}')

with col_ds:
    with st.container(border=True):
        st.markdown('##### For the data science team')
        if ds.get('summary'):
            st.markdown(ds['summary'])

        levers = ds.get('levers', [])
        if levers:
            st.markdown('**Improvement levers:**')
            for bucket in ['feature_engineering', 'model_param', 'workflow', 'algorithm']:
                bucket_levers = [lv for lv in levers if lv.get('bucket') == bucket]
                if not bucket_levers:
                    continue
                st.markdown(f'**{_BUCKET_LABEL.get(bucket, bucket)}**')
                for lv in bucket_levers:
                    effort_tag = _EFFORT_LABEL.get(lv.get('effort', ''), '')
                    change = lv.get('change', '')
                    st.markdown(f'- {change}' + (f' *({effort_tag})*' if effort_tag else ''))
                    if lv.get('evidence'):
                        st.caption(f"Evidence: {lv['evidence']}")
                    if lv.get('expected_effect'):
                        st.caption(f"Expected: {lv['expected_effect']}")

        # Fallback for old format (recommended_actions field)
        elif ds.get('recommended_actions'):
            st.markdown('**Recommended actions:**')
            for a in ds.get('recommended_actions', []):
                st.markdown(f'- {a}')

st.caption(
    f'Generated by {insights.get("model_flash", "?")} (analysis) + '
    f'{insights.get("model_critic", "?")} (critic) · {insights.get("created_at", "?")[:10]}'
)


# ══════════════════════════════════════════════════════════════════════════════
# 6. TECHNICAL EVIDENCE (off by default)
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
show_tech = st.toggle('🔬 Show the technical evidence (findings ledger + per-item XAI)', value=False)

if show_tech:
    # ── Findings ledger ────────────────────────────────────────────────────────
    st.markdown('#### Findings ledger')
    st.caption('Every candidate finding, with the critic\'s verdict. Fully traceable to data.')

    ledger_df = pd.DataFrame([
        {
            'Finding': f['finding_id'],
            'Type': f['finding_type'],
            'Status': f['status'],
            'Confidence': f['confidence'],
            'Critic notes (preview)': (f['critic_notes'] or '')[:120],
        }
        for f in findings
    ])
    st.dataframe(ledger_df, width='stretch')

    selected_finding_id = st.selectbox(
        'Inspect a finding',
        options=[f['finding_id'] for f in findings],
        format_func=lambda fid: next(
            (f"{f['finding_id']} [{f['status']} / {f['confidence']}]"
             for f in findings if f['finding_id'] == fid), fid
        ),
    )
    selected_finding = next((f for f in findings if f['finding_id'] == selected_finding_id), None)

    if selected_finding:
        st.markdown(
            f"**Status:** {_status_badge(selected_finding['status'])}  "
            f"**Confidence:** {_confidence_badge(selected_finding['confidence'])}"
        )
        st.markdown(f"**Critic notes:** {selected_finding['critic_notes']}")

        if selected_finding.get('hypothesis'):
            hyp = selected_finding['hypothesis']
            exp = hyp.get('explanation', {})
            if isinstance(exp, str):
                try:
                    exp = json.loads(exp)
                except Exception:
                    exp = {'raw': exp}
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.markdown('**Data scientist explanation:**')
                st.markdown(exp.get('ds_explanation', ''))
                fix = exp.get('suggested_fix', '')
                if fix:
                    st.markdown(f'*Suggested fix:* {fix}')
            with col_h2:
                st.markdown('**Business explanation:**')
                st.markdown(exp.get('business_explanation', ''))

        with st.expander('Raw evidence (JSON)'):
            st.json(selected_finding.get('evidence', {}))

    # ── Per-item XAI drill-down ────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('#### Per-item XAI drill-down')
    st.caption('Pick a bad week and item to see SHAP, counterfactual, and contrastive explanations.')

    try:
        evals = _evaluations()
    except Exception:
        st.warning('No evaluation data.')
        evals = pd.DataFrame()

    bad_weeks = sorted(evals[evals['is_bad_week'] == 1]['week_id'].unique(), reverse=True) if not evals.empty else []
    if not bad_weeks:
        st.info('No bad weeks found in evaluations.')
    else:
        col_w, col_i = st.columns(2)
        selected_week = col_w.selectbox('Bad week', bad_weeks)
        week_items = evals[evals['week_id'] == selected_week].sort_values('h1_mape', ascending=False)
        selected_item = col_i.selectbox('Item (sorted by error)', week_items['item_id'].tolist())

        xai = _xai(selected_week, selected_item)
        if not xai:
            st.info('No XAI data for this item — only the top 50 worst items per bad week are explained.')
        else:
            item_row = week_items[week_items['item_id'] == selected_item].iloc[0]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric('Error (MAPE)', f"{item_row['h1_mape']:.1f}%")
            mc2.metric('MAE', f"{item_row['h1_mae']:.1f}")
            mc3.metric('Error spike (z-score)', f"{item_row['mape_zscore']:.2f}")

            tabs = st.tabs(['SHAP', 'Counterfactual', 'Contrastive'])

            # ── SHAP ───────────────────────────────────────────────────────────
            with tabs[0]:
                if 'shap' not in xai:
                    st.info('No SHAP data for this item.')
                else:
                    d = xai['shap']
                    s1, s2, s3, _ = st.columns(4)
                    s1.metric('Prediction', f"{d['prediction']:.1f}")
                    if d.get('actual') is not None:
                        s2.metric('Actual', f"{d['actual']:.1f}")
                    if d.get('error_pct') is not None:
                        direction = d.get('direction', '')
                        s3.metric('Error', f"{d['error_pct']:.1f}%",
                                  delta=f"{direction}-forecast", delta_color='inverse')
                    feats = d['top_features']
                    labels = [f"{f['feature']} = {f['feature_value']:.2f}" for f in feats]
                    values = [f['shap_value'] for f in feats]
                    base_log = d['base_value_log']
                    other_shap = d.get('other_features_shap')
                    n_other = len(FEATURE_COLS) - len(feats)
                    if other_shap is not None:
                        labels = labels + [f'other {n_other} features']
                        values = values + [other_shap]
                    measures = ['relative'] * len(labels) + ['total']
                    fig_s = go.Figure(go.Waterfall(
                        orientation='h',
                        measure=measures,
                        y=labels + ['log-margin total'],
                        x=values + [0],
                        base=base_log,
                        connector={'line': {'color': '#ccc'}},
                        increasing={'marker': {'color': ACCENT_OVER}},
                        decreasing={'marker': {'color': ACCENT_UNDER}},
                    ))
                    fig_s.update_layout(
                        title='SHAP waterfall — top 5 features + residual',
                        xaxis_title='Log-margin contribution',
                        height=360, margin=dict(l=200),
                    )
                    st.plotly_chart(fig_s, width='stretch')
                    st.caption(
                        f"Base log-margin: {base_log:.3f}. SHAP values in log space (Tweedie). "
                        'Red = pushes prediction up, blue = down.'
                    )

            # ── Counterfactual ─────────────────────────────────────────────────
            with tabs[1]:
                if 'counterfactual' not in xai:
                    st.info('No counterfactual data.')
                else:
                    d = xai['counterfactual']
                    cc1, cc2 = st.columns(2)
                    cc1.metric('Original prediction', f"{d['prediction_original']:.1f}")
                    if d.get('actual') is not None:
                        cc2.metric('Actual', f"{d['actual']:.1f}")
                    st.markdown(
                        'What if SNAP / event / price-change were absent this week? '
                        'Inactive scenarios (feature already zero) are shown for completeness.'
                    )
                    cf_rows = []
                    for s in d['scenarios']:
                        cf_rows.append({
                            'Scenario': s['scenario'],
                            'Active': 'Yes' if s.get('was_active') else 'No (inactive)',
                            'CF prediction': s['prediction_cf'],
                            'Delta': s['delta'],
                            'Delta %': s['delta_pct'],
                        })
                    active_df = pd.DataFrame([r for r in cf_rows if r['Active'] == 'Yes'])
                    if not active_df.empty:
                        fig_c = px.bar(
                            active_df, x='Scenario', y='Delta',
                            title='Prediction change (active scenarios only)',
                            color='Delta', color_continuous_scale='RdBu', color_continuous_midpoint=0,
                            text='Delta',
                        )
                        fig_c.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                        fig_c.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig_c, width='stretch')
                    else:
                        st.info('All features inactive this week — no meaningful counterfactuals.')
                    st.dataframe(
                        pd.DataFrame(cf_rows),
                        width='stretch',
                        column_config={
                            'Delta': st.column_config.NumberColumn(format='%.2f'),
                            'Delta %': st.column_config.NumberColumn(format='%.1f%%'),
                            'CF prediction': st.column_config.NumberColumn(format='%.2f'),
                        },
                    )

            # ── Contrastive ────────────────────────────────────────────────────
            with tabs[2]:
                if 'contrastive' not in xai:
                    st.info(
                        'No contrastive data — no same-week-of-year good reference found for this SKU. '
                        '(Most items have no qualifying reference week.)'
                    )
                else:
                    d = xai['contrastive']
                    st.markdown(
                        f"**Bad week:** {d['bad_week']}  ↔  "
                        f"**Good reference week:** {d['good_week']} "
                        f"(error {d['good_week_mape']:.1f}%)"
                        + ('  ✓ same week-of-year' if d.get('seasonality_matched') else '')
                    )
                    diffs_df = pd.DataFrame(d['top_diffs'])
                    fig_d = go.Figure()
                    fig_d.add_trace(go.Bar(
                        name='Bad week', x=diffs_df['feature'], y=diffs_df['bad_shap'],
                        marker_color=ACCENT_OVER,
                    ))
                    fig_d.add_trace(go.Bar(
                        name='Good week', x=diffs_df['feature'], y=diffs_df['good_shap'],
                        marker_color=ACCENT_UNDER,
                    ))
                    fig_d.update_layout(
                        barmode='group',
                        title='SHAP values: bad week vs good reference week',
                        yaxis_title='SHAP contribution (log-margin)', height=320,
                    )
                    st.plotly_chart(fig_d, width='stretch')
                    st.dataframe(
                        diffs_df[['feature', 'bad_value', 'good_value', 'bad_shap', 'good_shap', 'shap_diff']],
                        width='stretch',
                        column_config={
                            'shap_diff': st.column_config.NumberColumn('SHAP diff', format='%.3f'),
                            'bad_shap':  st.column_config.NumberColumn('Bad SHAP',  format='%.3f'),
                            'good_shap': st.column_config.NumberColumn('Good SHAP', format='%.3f'),
                            'bad_value': st.column_config.NumberColumn('Bad value',  format='%.2f'),
                            'good_value': st.column_config.NumberColumn('Good value', format='%.2f'),
                        },
                    )
