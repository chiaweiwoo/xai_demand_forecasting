"""
XAI Demand Forecasting — single-page dashboard.

Layout (top → bottom):
  1. MAPE time series + bad-week markers
  2. Two-perspective insights summary (data scientist | business leader)
  3. Findings ledger — pick a finding to see its evidence + hypothesis
  4. XAI drill-down — pick a bad week + item → SHAP / counterfactual / contrastive

Usage:
    uv run streamlit run app.py
"""

import json
from collections import defaultdict

import numpy as np
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

st.set_page_config(page_title='XAI Demand Forecasting', layout='wide')


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def _week_summary():
    with get_conn(DB_PATH) as conn:
        return week_summary(conn)


@st.cache_data
def _evaluations():
    with get_conn(DB_PATH) as conn:
        return load_evaluations(conn)


@st.cache_data
def _insight_summary():
    with get_conn(DB_PATH) as conn:
        return load_insight_summary(conn)


@st.cache_data
def _insight_findings():
    with get_conn(DB_PATH) as conn:
        return load_insight_findings(conn)


def _xai(week_id: str, item_id: str) -> dict[str, dict]:
    with get_conn(DB_PATH) as conn:
        rows = load_xai(conn, week_id, item_id)
    return {r['xai_type']: json.loads(r['payload']) for r in rows}


# ── Helpers ───────────────────────────────────────────────────────────────────

_CONF_BADGE = {'high': ':green[High]', 'medium': ':orange[Medium]', 'low': ':red[Low]'}
_STATUS_BADGE = {
    'accepted':     ':green[Accepted]',
    'rejected':     ':red[Rejected]',
    'needs_review': ':orange[Needs review]',
}


def _confidence_badge(conf: str) -> str:
    return _CONF_BADGE.get(conf, conf)


def _status_badge(status: str) -> str:
    return _STATUS_BADGE.get(status, status)


# ── Page title ────────────────────────────────────────────────────────────────

st.title('XAI Demand Forecasting')
st.caption('"The model performed badly at week X — why?"')


# ══════════════════════════════════════════════════════════════════════════════
# 1. MAPE time series
# ══════════════════════════════════════════════════════════════════════════════

st.header('Model Performance')

try:
    summary = _week_summary()
except Exception:
    st.warning('No data yet. Run the pipeline first: backtest.py → run_xai.py → generate_insights.py')
    st.stop()

summary['week_dt'] = pd.to_datetime(summary['week_id'])
bad = summary[summary['n_bad_items'] > 0]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=summary['week_dt'], y=summary['avg_mape'],
    mode='lines', name='Avg MAPE', line=dict(color='steelblue', width=1.5),
))
fig.add_trace(go.Scatter(
    x=bad['week_dt'], y=bad['avg_mape'],
    mode='markers', name='Bad week',
    marker=dict(color='tomato', size=9, symbol='x-thin', line=dict(width=2)),
))
fig.update_layout(
    xaxis_title='Week', yaxis_title='Avg MAPE (%)',
    hovermode='x unified', height=320,
    margin=dict(t=20),
)
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Total weeks', len(summary))
c2.metric('Bad weeks', int(bad['n_bad_items'].gt(0).sum()) if len(bad) else 0)
c3.metric('Overall avg MAPE', f"{summary['avg_mape'].mean():.1f}%")
c4.metric('Worst week MAPE', f"{summary['avg_mape'].max():.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Two-perspective insights summary
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header('Insights Summary')

insights = _insight_summary()
if not insights:
    st.info('No insights yet. Run: `uv run python generate_insights.py`')
else:
    overall_conf = insights.get('data_scientist', {}).get('overall_confidence') or \
                   insights.get('business_leader', {}).get('overall_confidence') or 'low'

    ds  = insights['data_scientist']
    biz = insights['business_leader']

    col_ds, col_biz = st.columns(2)

    with col_ds:
        st.subheader('Data Scientist View')
        st.markdown(f"**{ds.get('headline', '')}**")
        st.markdown(ds.get('summary', ''))

        issues = ds.get('top_issues', [])
        if issues:
            st.markdown('**Top issues:**')
            for issue in issues:
                st.markdown(f'- {issue}')

        actions = ds.get('recommended_actions', [])
        if actions:
            st.markdown('**Recommended actions:**')
            for action in actions:
                st.markdown(f'- {action}')

    with col_biz:
        st.subheader('Business Leader View')
        st.markdown(f"**{biz.get('headline', '')}**")
        st.markdown(biz.get('summary', ''))

        risk = biz.get('risk_direction', '')
        if risk:
            risk_color = 'red' if risk == 'over-stock' else ('orange' if risk == 'under-stock' else 'gray')
            st.markdown(f'**Risk direction:** :{risk_color}[{risk}]')

        limitations = biz.get('limitations', [])
        if limitations:
            st.markdown('**Known limitations:**')
            for lim in limitations:
                st.markdown(f'- {lim}')

        plan = biz.get('improvement_plan', '')
        if plan:
            st.markdown(f'**Improvement plan:** {plan}')

    st.caption(
        f'Models: Flash={insights.get("model_flash", "?")} · '
        f'Critic={insights.get("model_critic", "?")} · '
        f'Generated: {insights.get("created_at", "?")[:19]}'
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Findings ledger
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header('Findings Ledger')
st.caption('Each finding is traceable to real data. Critic (Pro model) accepted/rejected each one.')

findings = _insight_findings()
if not findings:
    st.info('No findings yet. Run: `uv run python generate_insights.py`')
else:
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
    st.dataframe(
        ledger_df,
        use_container_width=True,
        column_config={
            'Status': st.column_config.TextColumn(),
            'Confidence': st.column_config.TextColumn(),
        },
    )

    selected_finding_id = st.selectbox(
        'Inspect a finding',
        options=[f['finding_id'] for f in findings],
        format_func=lambda fid: next(
            (f"{f['finding_id']} [{f['status']} / {f['confidence']}]" for f in findings if f['finding_id'] == fid), fid
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


# ══════════════════════════════════════════════════════════════════════════════
# 4. XAI drill-down
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header('XAI Drill-Down')
st.caption('Pick a bad week and item to see SHAP, counterfactual, and contrastive explanations.')

try:
    evals = _evaluations()
except Exception:
    st.warning('No evaluation data. Run backtest.py first.')
    st.stop()

bad_weeks = sorted(evals[evals['is_bad_week'] == 1]['week_id'].unique(), reverse=True)
if not bad_weeks:
    st.info('No bad weeks found in evaluations.')
else:
    col_w, col_i = st.columns(2)
    selected_week = col_w.selectbox('Bad week', bad_weeks)
    week_items = evals[evals['week_id'] == selected_week].sort_values('h1_mape', ascending=False)
    selected_item = col_i.selectbox('Item (sorted by MAPE)', week_items['item_id'].tolist())

    xai = _xai(selected_week, selected_item)
    if not xai:
        st.info('No XAI data for this item — only the top 50 worst items per bad week are explained.')
    else:
        item_row = week_items[week_items['item_id'] == selected_item].iloc[0]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric('MAPE', f"{item_row['h1_mape']:.1f}%")
        mc2.metric('MAE', f"{item_row['h1_mae']:.1f}")
        mc3.metric('WMAPE z-score', f"{item_row['mape_zscore']:.2f}")

        tabs = st.tabs(['SHAP', 'Counterfactual', 'Contrastive'])

        # ── SHAP ─────────────────────────────────────────────────────────────
        with tabs[0]:
            if 'shap' not in xai:
                st.info('No SHAP data for this item.')
            else:
                d = xai['shap']
                s1, s2, s3, s4 = st.columns(4)
                s1.metric('Prediction', f"{d['prediction']:.1f}")
                if d.get('actual') is not None:
                    s2.metric('Actual', f"{d['actual']:.1f}")
                if d.get('error_pct') is not None:
                    direction = d.get('direction', '')
                    s3.metric('Error', f"{d['error_pct']:.1f}%", delta=f"{direction}-forecast", delta_color='inverse')
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
                fig = go.Figure(go.Waterfall(
                    orientation='h',
                    measure=measures,
                    y=labels + ['log-margin total'],
                    x=values + [0],
                    base=base_log,
                    connector={'line': {'color': '#ccc'}},
                    increasing={'marker': {'color': 'tomato'}},
                    decreasing={'marker': {'color': 'steelblue'}},
                ))
                fig.update_layout(
                    title='SHAP waterfall — top 5 features + residual',
                    xaxis_title='Log-margin contribution',
                    height=360, margin=dict(l=200),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"Base log-margin: {base_log:.3f}. SHAP values in log space (Tweedie). "
                    'Red = pushes prediction up, blue = down.'
                )

        # ── Counterfactual ───────────────────────────────────────────────────
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
                    fig = px.bar(
                        active_df, x='Scenario', y='Delta',
                        title='Prediction change (active scenarios only)',
                        color='Delta', color_continuous_scale='RdBu', color_continuous_midpoint=0,
                        text='Delta',
                    )
                    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info('All features inactive this week — no meaningful counterfactuals.')
                st.dataframe(
                    pd.DataFrame(cf_rows),
                    use_container_width=True,
                    column_config={
                        'Delta': st.column_config.NumberColumn(format='%.2f'),
                        'Delta %': st.column_config.NumberColumn(format='%.1f%%'),
                        'CF prediction': st.column_config.NumberColumn(format='%.2f'),
                    },
                )

        # ── Contrastive ──────────────────────────────────────────────────────
        with tabs[2]:
            if 'contrastive' not in xai:
                st.info(
                    'No contrastive data — no same-week-of-year good reference found for this SKU. '
                    '(73% of items have no qualifying reference week.)'
                )
            else:
                d = xai['contrastive']
                st.markdown(
                    f"**Bad week:** {d['bad_week']}  ↔  "
                    f"**Good reference week:** {d['good_week']} "
                    f"(MAPE {d['good_week_mape']:.1f}%)"
                    + ('  ✓ same week-of-year' if d.get('seasonality_matched') else '')
                )
                diffs_df = pd.DataFrame(d['top_diffs'])
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Bad week', x=diffs_df['feature'], y=diffs_df['bad_shap'],
                    marker_color='tomato',
                ))
                fig.add_trace(go.Bar(
                    name='Good week', x=diffs_df['feature'], y=diffs_df['good_shap'],
                    marker_color='steelblue',
                ))
                fig.update_layout(
                    barmode='group',
                    title='SHAP values: bad week vs good reference week',
                    yaxis_title='SHAP contribution (log-margin)', height=320,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    diffs_df[['feature', 'bad_value', 'good_value', 'bad_shap', 'good_shap', 'shap_diff']],
                    use_container_width=True,
                    column_config={
                        'shap_diff': st.column_config.NumberColumn('SHAP diff', format='%.3f'),
                        'bad_shap':  st.column_config.NumberColumn('Bad SHAP',  format='%.3f'),
                        'good_shap': st.column_config.NumberColumn('Good SHAP', format='%.3f'),
                        'bad_value': st.column_config.NumberColumn('Bad value',  format='%.2f'),
                        'good_value': st.column_config.NumberColumn('Good value', format='%.2f'),
                    },
                )
