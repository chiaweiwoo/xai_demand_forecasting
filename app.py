"""
Streamlit dashboard for XAI demand forecasting.

Usage:
    uv run streamlit run app.py
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from xai_forecast.db import (
    get_conn, week_summary, load_evaluations, load_xai, load_all_shap_payloads,
    load_narrative,
)

DB_PATH = 'db/forecasting.db'

st.set_page_config(page_title='XAI Demand Forecasting', layout='wide')


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def _week_summary():
    with get_conn(DB_PATH) as conn:
        return week_summary(conn)


@st.cache_data
def _evaluations():
    with get_conn(DB_PATH) as conn:
        return load_evaluations(conn)


def _xai(week_id: str, item_id: str) -> dict[str, dict]:
    with get_conn(DB_PATH) as conn:
        rows = load_xai(conn, week_id, item_id)
    return {r['xai_type']: json.loads(r['payload']) for r in rows}


def _week_xai(week_id: str) -> list[dict]:
    """All XAI rows for a bad week (all items, all types)."""
    with get_conn(DB_PATH) as conn:
        return load_xai(conn, week_id)


@st.cache_data
def _all_shap_payloads() -> list[dict]:
    with get_conn(DB_PATH) as conn:
        return load_all_shap_payloads(conn)


def _week_shap_summary(week_id: str) -> pd.DataFrame:
    """
    Mean |SHAP| per feature across all SHAP payloads for a bad week.
    Only features that appeared in the top-5 for at least one SKU are shown.
    """
    rows = _week_xai(week_id)
    feature_scores: dict[str, list[float]] = defaultdict(list)
    n_skus = 0
    for r in rows:
        if r['xai_type'] != 'shap':
            continue
        n_skus += 1
        p = json.loads(r['payload'])
        for f in p.get('top_features', []):
            feature_scores[f['feature']].append(abs(f['shap_value']))
    if not feature_scores:
        return pd.DataFrame()
    return pd.DataFrame([
        {'feature': feat, 'mean_abs_shap': float(np.mean(vals)),
         'n_skus': len(vals), 'pct_skus': len(vals) / n_skus * 100 if n_skus else 0}
        for feat, vals in feature_scores.items()
    ]).sort_values('mean_abs_shap', ascending=False)


@st.cache_data
def _recurring_drivers() -> pd.DataFrame:
    """
    Across all bad weeks, how often does each feature appear in the top-5 SHAP drivers?
    Returns: feature → count (total appearances across all SHAP payloads).
    """
    all_rows = _all_shap_payloads()
    feature_counts: dict[str, int] = defaultdict(int)
    total_payloads = 0
    for r in all_rows:
        p = json.loads(r['payload'])
        total_payloads += 1
        for f in p.get('top_features', []):
            feature_counts[f['feature']] += 1
    if not feature_counts:
        return pd.DataFrame()
    df = pd.DataFrame([
        {'feature': feat, 'count': cnt,
         'pct_payloads': cnt / total_payloads * 100 if total_payloads else 0}
        for feat, cnt in feature_counts.items()
    ]).sort_values('count', ascending=False)
    df['total_payloads'] = total_payloads
    return df


def _week_narrative(week_id: str) -> dict | None:
    try:
        with get_conn(DB_PATH) as conn:
            return load_narrative(conn, 'week', week_id)
    except Exception:
        return None


def _item_narrative(week_id: str, item_id: str) -> dict | None:
    try:
        with get_conn(DB_PATH) as conn:
            return load_narrative(conn, 'item', f'{week_id}::{item_id}')
    except Exception:
        return None


@st.cache_data
def _executive_narrative() -> dict | None:
    try:
        with get_conn(DB_PATH) as conn:
            return load_narrative(conn, 'executive', 'overall')
    except Exception:
        return None


def _render_narrative(narr: dict) -> None:
    """Render a narrative dict as a styled info box."""
    conf = narr.get('confidence', '')
    badge = {'high': '🟢 High', 'medium': '🟡 Medium', 'low': '🔴 Low'}.get(conf, conf)
    grounding_note = ' *(grounding check flagged — verify manually)*' if narr.get('grounding_warning') else ''
    st.info(
        f"**{narr.get('headline', '')}**\n\n"
        f"{narr.get('body', '')}\n\n"
        f"*Primary driver: `{narr.get('primary_driver', '?')}` · "
        f"Confidence: {badge}{grounding_note}*"
    )


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title('XAI Demand Forecasting')
page = st.sidebar.radio('View', ['Overview', 'Bad Week Drilldown', 'Recurring Drivers', 'XAI Explorer'])


# ── Overview ──────────────────────────────────────────────────────────────────

if page == 'Overview':
    st.title('Model Performance Overview')

    try:
        summary = _week_summary()
    except Exception:
        st.warning('No data yet. Run: `uv run python backtest.py`')
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
        title='Weekly h=1 Forecast Error',
        hovermode='x unified', height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total weeks', len(summary))
    c2.metric('Bad weeks', int(summary['n_bad_items'].gt(0).sum()))
    c3.metric('Overall avg MAPE', f"{summary['avg_mape'].mean():.1f}%")
    c4.metric('Worst week MAPE', f"{summary['avg_mape'].max():.1f}%")


# ── Bad Week Drilldown ────────────────────────────────────────────────────────

elif page == 'Bad Week Drilldown':
    st.title('Bad Week Drilldown')

    try:
        evals = _evaluations()
    except Exception:
        st.warning('No data yet. Run: `uv run python backtest.py`')
        st.stop()

    bad_weeks = sorted(evals[evals['is_bad_week'] == 1]['week_id'].unique(), reverse=True)
    if not bad_weeks:
        st.info('No bad weeks found.')
        st.stop()

    selected = st.selectbox('Select a bad week', bad_weeks)
    week_df = evals[evals['week_id'] == selected].sort_values('h1_mape', ascending=False)

    narr = _week_narrative(selected)
    if narr:
        _render_narrative(narr)

    c1, c2, c3 = st.columns(3)
    c1.metric('Items tracked', len(week_df))
    c2.metric('Avg MAPE', f"{week_df['h1_mape'].mean():.1f}%")
    c3.metric('MAPE z-score', f"{week_df['mape_zscore'].iloc[0]:.2f}")

    st.subheader('Worst 30 items')
    st.dataframe(
        week_df[['item_id', 'h1_mape', 'h1_mae', 'mape_zscore']].head(30).reset_index(drop=True),
        use_container_width=True,
        column_config={
            'h1_mape': st.column_config.NumberColumn('MAPE (%)', format='%.1f'),
            'h1_mae': st.column_config.NumberColumn('MAE', format='%.1f'),
            'mape_zscore': st.column_config.NumberColumn('Z-score', format='%.2f'),
        },
    )

    # ── Week-level SHAP driver summary ────────────────────────────────────────
    st.subheader('What drove the bad week? (aggregated SHAP across top-50 SKUs)')
    shap_summary = _week_shap_summary(selected)
    if shap_summary.empty:
        st.info('No SHAP data for this week. Run backtest to generate XAI.')
    else:
        fig_shap = go.Figure(go.Bar(
            x=shap_summary['mean_abs_shap'],
            y=shap_summary['feature'],
            orientation='h',
            text=shap_summary['pct_skus'].apply(lambda x: f'{x:.0f}% of SKUs'),
            textposition='outside',
            marker_color='steelblue',
        ))
        fig_shap.update_layout(
            title='Mean |SHAP| per feature across worst SKUs this week (log-margin space)',
            xaxis_title='Mean |SHAP|',
            yaxis=dict(autorange='reversed'),
            height=max(300, len(shap_summary) * 28),
            margin=dict(l=160),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            'Only features that appeared in the top-5 for at least one SKU are shown. '
            '"% of SKUs" = fraction of the top-50 for which this feature was a top driver.'
        )

    st.subheader('MAPE distribution this week')
    fig = px.histogram(week_df, x='h1_mape', nbins=40, title='Item MAPE distribution',
                       color_discrete_sequence=['steelblue'])
    st.plotly_chart(fig, use_container_width=True)


# ── Recurring Drivers ─────────────────────────────────────────────────────────

elif page == 'Recurring Drivers':
    st.title('Recurring Failure Drivers')
    st.caption(
        'Across **all** bad weeks, which features most often appear in the top-5 SHAP drivers '
        'for the worst-performing SKUs? High frequency = systematic model blind spot.'
    )

    try:
        drivers = _recurring_drivers()
    except Exception:
        st.warning('No data yet. Run: `uv run python backtest.py`')
        st.stop()

    if drivers.empty:
        st.info('No SHAP data found. Run backtest.py to generate XAI results.')
        st.stop()

    exec_narr = _executive_narrative()
    if exec_narr:
        _render_narrative(exec_narr)

    total = int(drivers['total_payloads'].iloc[0])
    st.markdown(f'Based on **{total:,}** SHAP explanations across all bad weeks.')

    fig = go.Figure(go.Bar(
        x=drivers['count'],
        y=drivers['feature'],
        orientation='h',
        text=drivers['pct_payloads'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        marker_color=px.colors.sequential.Blues_r[:len(drivers)],
    ))
    fig.update_layout(
        title='Feature appearance frequency in top-5 SHAP drivers (all bad weeks)',
        xaxis_title='Count (appearances across all bad-week SHAP payloads)',
        yaxis=dict(autorange='reversed'),
        height=max(350, len(drivers) * 28),
        margin=dict(l=160),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Feature frequency table')
    st.dataframe(
        drivers[['feature', 'count', 'pct_payloads']].rename(
            columns={'count': 'Appearances', 'pct_payloads': '% of payloads'}
        ).reset_index(drop=True),
        use_container_width=True,
        column_config={
            '% of payloads': st.column_config.NumberColumn(format='%.1f%%'),
        },
    )
    st.caption(
        'A feature appearing in 80% of bad-week payloads is likely a systematic driver. '
        'Investigate whether it reflects real-world signal or a structural model weakness.'
    )


# ── XAI Explorer ─────────────────────────────────────────────────────────────

elif page == 'XAI Explorer':
    st.title('XAI Explorer')
    st.caption('Leader asks: "Why did the model fail at week X?" — pick a bad week and item.')

    try:
        evals = _evaluations()
    except Exception:
        st.warning('No data yet. Run: `uv run python backtest.py`')
        st.stop()

    bad_weeks = sorted(evals[evals['is_bad_week'] == 1]['week_id'].unique(), reverse=True)
    if not bad_weeks:
        st.info('No bad weeks found.')
        st.stop()

    col1, col2 = st.columns(2)
    selected_week = col1.selectbox('Bad week', bad_weeks)
    week_items = evals[evals['week_id'] == selected_week].sort_values('h1_mape', ascending=False)
    selected_item = col2.selectbox('Item (sorted by MAPE ↓)', week_items['item_id'].tolist())

    xai = _xai(selected_week, selected_item)
    if not xai:
        st.info('No XAI data for this item. Only the top 50 worst items per bad week are explained.')
        st.stop()

    item_narr = _item_narrative(selected_week, selected_item)
    if item_narr:
        _render_narrative(item_narr)

    tabs = st.tabs(['SHAP', 'Counterfactual', 'Contrastive'])

    # ── SHAP ─────────────────────────────────────────────────────────────────
    with tabs[0]:
        if 'shap' not in xai:
            st.info('No SHAP data.')
        else:
            d = xai['shap']
            c1, c2, c3 = st.columns(3)
            c1.metric('Model prediction', f"{d['prediction']:.1f}")
            if d.get('actual') is not None:
                c2.metric('Actual sales', f"{d['actual']:.1f}")
            if d.get('error_pct') is not None:
                c3.metric('Error', f"{d['error_pct']:.1f}%")

            feats = d['top_features']
            labels = [f"{f['feature']} = {f['feature_value']:.2f}" for f in feats]
            values = [f['shap_value'] for f in feats]
            base_log = d['base_value_log']

            other_shap = d.get('other_features_shap')
            n_other = 19 - len(feats)
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
                title='SHAP waterfall — top 5 features + residual (log-margin contributions)',
                xaxis_title='Log-margin contribution',
                height=380, margin=dict(l=200),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Base log-margin: {base_log:.3f} (= log of average prediction). "
                'SHAP values are additive in log space (Tweedie log-link). '
                'Red bars push the log-prediction up, blue push it down. '
                'The "other N features" bar shows the residual so the waterfall reconciles to the prediction.'
            )

    # ── Counterfactual ───────────────────────────────────────────────────────
    with tabs[1]:
        if 'counterfactual' not in xai:
            st.info('No counterfactual data.')
        else:
            d = xai['counterfactual']
            c1, c2 = st.columns(2)
            c1.metric('Original prediction', f"{d['prediction_original']:.1f}")
            if d.get('actual') is not None:
                c2.metric('Actual', f"{d['actual']:.1f}")

            st.markdown(
                '**What if we remove each feature?** '
                'A large negative delta means the feature inflated the prediction. '
                'Grayed-out scenarios are features already inactive for this SKU this week — '
                'zeroing them has no effect by definition.'
            )

            scenarios = d['scenarios']
            rows_cf = []
            for s in scenarios:
                active = s.get('was_active', True)
                rows_cf.append({
                    'Scenario': s['scenario'],
                    'Active this week': '✓' if active else '✗ (inactive)',
                    'CF prediction': s['prediction_cf'],
                    'Delta': s['delta'],
                    'Delta %': s['delta_pct'],
                })
            rows_df = pd.DataFrame(rows_cf)

            # Only plot active scenarios in the bar chart (inactive delta is trivially ~0)
            active_rows = rows_df[rows_df['Active this week'] == '✓']
            if not active_rows.empty:
                fig = px.bar(
                    active_rows, x='Scenario', y='Delta',
                    title='Prediction change when feature is zeroed out (active scenarios only)',
                    color='Delta',
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0,
                    text='Delta',
                )
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('All features were inactive for this SKU this week.')

            st.dataframe(
                rows_df,
                use_container_width=True,
                column_config={
                    'Delta': st.column_config.NumberColumn(format='%.2f'),
                    'Delta %': st.column_config.NumberColumn(format='%.1f%%'),
                    'CF prediction': st.column_config.NumberColumn(format='%.2f'),
                },
            )

    # ── Contrastive ──────────────────────────────────────────────────────────
    with tabs[2]:
        if 'contrastive' not in xai:
            st.info('No contrastive data (no same-week-of-year good reference found for this SKU).')
        else:
            d = xai['contrastive']
            matched = d.get('seasonality_matched', False)
            st.markdown(
                f"**Bad week:** {d['bad_week']}  ↔  "
                f"**Reference good week:** {d['good_week']} "
                f"(MAPE {d['good_week_mape']:.1f}%)"
                + ('  ✓ same week-of-year' if matched else '')
            )
            st.caption(
                'Comparing SHAP values between this bad week and a historical week '
                'with the same ISO week-of-year where the model performed well. '
                'Features with large SHAP differences are the structural divergence between the two weeks.'
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
                yaxis_title='SHAP contribution (log-margin)', height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                diffs_df[['feature', 'bad_value', 'good_value', 'bad_shap', 'good_shap', 'shap_diff']],
                use_container_width=True,
                column_config={
                    'shap_diff': st.column_config.NumberColumn('SHAP diff', format='%.3f'),
                    'bad_shap': st.column_config.NumberColumn('Bad SHAP', format='%.3f'),
                    'good_shap': st.column_config.NumberColumn('Good SHAP', format='%.3f'),
                    'bad_value': st.column_config.NumberColumn('Bad value', format='%.2f'),
                    'good_value': st.column_config.NumberColumn('Good value', format='%.2f'),
                },
            )
