"""
Streamlit dashboard for XAI demand forecasting.

Usage:
    uv run streamlit run app.py
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from xai_forecast.db import get_conn, week_summary, load_evaluations, load_xai

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


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title('XAI Demand Forecasting')
page = st.sidebar.radio('View', ['Overview', 'Bad Week Drilldown', 'XAI Explorer'])


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

    st.subheader('MAPE distribution this week')
    fig = px.histogram(week_df, x='h1_mape', nbins=40, title='Item MAPE distribution',
                       color_discrete_sequence=['steelblue'])
    st.plotly_chart(fig, use_container_width=True)


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
    selected_week = col1.selectbox('Bad week (cutoff)', bad_weeks)
    week_items = evals[evals['week_id'] == selected_week].sort_values('h1_mape', ascending=False)
    selected_item = col2.selectbox('Item (sorted by MAPE ↓)', week_items['item_id'].tolist())

    xai = _xai(selected_week, selected_item)
    if not xai:
        st.info('No XAI data for this item. Only the top 50 worst items per bad week are explained.')
        st.stop()

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

            fig = go.Figure(go.Waterfall(
                orientation='h',
                measure=['relative'] * len(feats) + ['total'],
                y=labels + ['final prediction'],
                x=values + [0],
                base=d['base_value'],
                connector={'line': {'color': '#ccc'}},
                increasing={'marker': {'color': 'tomato'}},
                decreasing={'marker': {'color': 'steelblue'}},
            ))
            fig.update_layout(
                title='SHAP waterfall — top 5 features driving the prediction',
                height=340, margin=dict(l=200),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Base value (average prediction): {d['base_value']:.2f}. "
                'Red bars push the prediction up, blue push it down.'
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
                'A large negative delta means the feature inflated the prediction '
                '(possibly explaining why the model over-predicted).'
            )
            rows_cf = pd.DataFrame(d['scenarios'])
            rows_cf.columns = ['Scenario', 'CF prediction', 'Delta', 'Delta %']

            fig = px.bar(
                rows_cf, x='Scenario', y='Delta',
                title='Prediction change when feature is zeroed out',
                color='Delta',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                text='Delta',
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rows_cf, use_container_width=True)

    # ── Contrastive ──────────────────────────────────────────────────────────
    with tabs[2]:
        if 'contrastive' not in xai:
            st.info('No contrastive data (no matching good week found).')
        else:
            d = xai['contrastive']
            st.markdown(
                f"**Bad week:** {d['bad_week']}  ↔  "
                f"**Reference good week:** {d['good_week']} "
                f"(MAPE {d['good_week_mape']:.1f}%)"
            )
            st.caption(
                'Comparing SHAP values between this bad week and a historical week '
                'with similar seasonality where the model performed well. '
                'Features with large SHAP differences are the structural divergence.'
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
                yaxis_title='SHAP contribution', height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                diffs_df[['feature', 'bad_value', 'good_value', 'bad_shap', 'good_shap', 'shap_diff']],
                use_container_width=True,
                column_config={
                    'shap_diff': st.column_config.NumberColumn('SHAP diff', format='%.3f'),
                    'bad_shap': st.column_config.NumberColumn('Bad SHAP', format='%.3f'),
                    'good_shap': st.column_config.NumberColumn('Good SHAP', format='%.3f'),
                },
            )
