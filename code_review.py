"""
Code Review & Walkthrough — XAI Demand Forecasting
Guided tour of every module: how the ML, XAI, and LLM layers connect.

Run:
    uv run streamlit run code_review.py
"""

import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent

st.set_page_config(
    page_title='Code Review — XAI Demand Forecasting',
    layout='wide',
    page_icon='🔬',
    initial_sidebar_state='expanded',
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read(rel: str) -> str:
    try:
        return (ROOT / rel).read_text(encoding='utf-8')
    except FileNotFoundError:
        return f'# File not found: {rel}'


def _excerpt(src: str, start: int, end: int) -> str:
    return '\n'.join(src.splitlines()[start - 1:end])


def _header(title: str, subtitle: str, color: str) -> None:
    st.markdown(
        f'<div style="background:{color};padding:18px 22px;border-radius:10px;margin-bottom:16px">'
        f'<h2 style="color:#fff;margin:0;font-size:1.4rem">{title}</h2>'
        f'<p style="color:rgba(255,255,255,0.85);margin:5px 0 0;font-size:0.9rem">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _note(text: str, border: str = '#4a7fe3') -> None:
    st.markdown(
        f'<div style="border-left:4px solid {border};background:#f8f9fa;padding:10px 14px;'
        f'border-radius:0 6px 6px 0;margin:10px 0;font-size:0.88rem">{text}</div>',
        unsafe_allow_html=True,
    )


def _file_badge(path: str) -> None:
    st.markdown(
        f'<code style="background:#e8eaf6;color:#3949ab;padding:3px 8px;border-radius:4px;font-size:0.8rem">📄 {path}</code>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title('🔬 Code Review')
st.sidebar.caption('XAI Demand Forecasting — Guided walkthrough')
st.sidebar.markdown('---')

SECTIONS = [
    '🗺️  Overview',
    '📦  Data Layer',
    '🤖  ML Pipeline',
    '🔍  XAI Engine',
    '💬  Insights Module',
    '⚙️  Orchestration',
    '🗄️  Storage',
    '📊  Dashboard',
]
section = st.sidebar.radio('Navigate', SECTIONS, label_visibility='collapsed')

COLORS = {
    '🗺️  Overview':       '#1a73e8',
    '📦  Data Layer':     '#1565c0',
    '🤖  ML Pipeline':    '#2e7d32',
    '🔍  XAI Engine':     '#e65100',
    '💬  Insights Module':'#6a1b9a',
    '⚙️  Orchestration':  '#37474f',
    '🗄️  Storage':        '#00695c',
    '📊  Dashboard':      '#0277bd',
}

st.sidebar.markdown('---')
st.sidebar.markdown(
    '**Layer map**\n'
    '- 📦 Data → features table\n'
    '- 🤖 ML → forecasts + bad-week flags\n'
    '- 🔍 XAI → SHAP / CF / contrastive\n'
    '- 💬 LLM → insights (detectors → planner → critic)\n'
    '- 📊 Dashboard → single-page insights view'
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if section == '🗺️  Overview':
    _header(
        '🗺️ Project Overview',
        'Retrospective XAI on M5 (Walmart) weekly demand — answering the leader\'s question.',
        COLORS[section],
    )

    st.markdown(
        '### Core question\n'
        '> **"The model performed badly at week X — why?"**\n\n'
        'Standard dashboards show a MAPE spike and stop there. '
        'This project goes further: for every bad week it produces three types of XAI explanation '
        'for all valid non-pre-launch SKUs, then synthesises them into plain-English insights a business '
        'leader can read without knowing what SHAP is.'
    )

    st.markdown('---')
    st.markdown('### Pipeline flow')
    st.markdown(
        '''
<div style="font-family:monospace;font-size:0.82rem;line-height:2;background:#f8f9fa;padding:16px 20px;border-radius:8px">

<b style="color:#1565c0">DATA LAYER</b><br>
&nbsp;&nbsp;M5 CSVs (Kaggle) <br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>ingest.py</b> → SQLite: <i>weekly_sales, calendar, prices, item_meta</i><br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>build_features.py</b> → SQLite: <i>features</i> (847k rows, 26 cols, ~46s one-time)<br>

<br><b style="color:#2e7d32">ML LAYER</b><br>
&nbsp;&nbsp;<b>backtest.py</b> ← feature store (SQL SELECT per week, ~2s)<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>train.py</b>: LightGBM Tweedie (retrain every 4 weeks, 3-year window, ×30 checkpoints)<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>forecast.py</b>: h=1 predictions per SKU<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>evaluate.py</b>: WMAPE z-score ≥ 1.5 → bad-week flag<br>

<br><b style="color:#e65100">XAI LAYER</b><br>
&nbsp;&nbsp;For each bad week (all valid non-pre-launch SKUs):<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>xai.py / SHAP</b>: TreeSHAP — top-5 drivers in log-margin space<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>xai.py / Counterfactual</b>: zero out SNAP/event/price → measure prediction delta<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>xai.py / Contrastive</b>: diff SHAP vs a good same-WOY reference week<br>

<br><b style="color:#6a1b9a">INSIGHTS LAYER</b><br>
&nbsp;&nbsp;<b>generate_insights.py</b>: deterministic detectors → LangGraph fan-out<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ per-finding: planner (Flash) → enrich evidence → hypothesis (Flash) → grounding advisory → critic (Pro)<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ fan-in: synthesis (Flash) → DS view + business view<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ Logs full agent trace to logs/insights.log<br>

<br><b style="color:#00695c">STORAGE</b><br>
&nbsp;&nbsp;<b>db.py</b>: SQLite (WAL mode) — 10 tables, schema auto-applied via 7 migrations<br>
&nbsp;&nbsp;All writes are INSERT OR REPLACE — pipeline is idempotent, re-runnable<br>

<br><b style="color:#0277bd">DASHBOARD</b><br>
&nbsp;&nbsp;<b>app.py</b>: Streamlit — single-page: MAPE chart · Insights summary · Findings ledger · XAI drill-down<br>
</div>
''',
        unsafe_allow_html=True,
    )

    st.markdown('---')
    st.markdown('### Key numbers')
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Store scope', 'CA_1 only', '~3,049 SKUs')
    c2.metric('Total weeks', '278', '2011–2016')
    c3.metric('Backtest weeks', '~120', 'sliding window')
    c4.metric('Retrains', '~30', 'every 4 weeks')
    c5.metric('Features', '26', '19 ML + 7 external signals')

    st.markdown('---')
    st.markdown('### Why these design choices?')
    with st.expander('Tweedie objective — not RMSE or MAE'):
        st.markdown(
            '64% of weekly sales rows are zero (intermittent demand). '
            'Standard RMSE treats zero-sales weeks the same as high-volume ones. '
            'Tweedie (variance_power=1.5) is the textbook distribution for zero-heavy count data — '
            'it uses a log-link internally, which is why SHAP values are in log-margin space.'
        )
    with st.expander('WMAPE z-score — not raw MAPE threshold'):
        st.markdown(
            'A fixed MAPE threshold (e.g. > 40%) would flag every single week on low-volume SKUs. '
            'WMAPE (volume-weighted) normalises by total sales so large SKUs dominate the signal. '
            'Z-score against a prior-8-week baseline means we only flag weeks that spike *relative to recent history*. '
            'The `shift(1)` on the baseline is critical: without it, a bad week\'s own WMAPE inflates its baseline, '
            'damping its own z-score.'
        )
    with st.expander('Per-checkpoint XAI — not one final model'):
        st.markdown(
            'The model was retrained ~30 times during the backtest. '
            'Explaining a 2013 bad week with the model trained in 2016 would produce wrong SHAP values — '
            'the model has seen data the 2013 version hadn\'t. '
            'Every bad week is explained using the exact checkpoint that produced its forecast.'
        )
    with st.expander('Feature store — not recompute per iteration'):
        st.markdown(
            'Without a feature store, every backtest iteration would recompute all 847k feature rows — '
            '~36s per iteration × ~120 iterations = impractical. '
            '`build_features.py` computes them once (~46s total). Backtest then does a SQL SELECT (~2s). '
            'The smoke test verifies the store is not stale before a full run.'
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '📦  Data Layer':
    _header('📦 Data Layer', 'features.py · build_features.py · ingest.py', COLORS[section])

    features_src = _read('xai_forecast/features.py')

    tab1, tab2 = st.tabs(['features.py — feature engineering', 'build_features.py — precompute'])

    with tab1:
        _file_badge('xai_forecast/features.py')
        st.markdown('**Single source of truth** for all 26 features. Used by `build_features.py`, `backtest.py`, `smoke_test.py`, and all tests.')

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('#### Feature columns (26 total)')
            st.code(_excerpt(features_src, 14, 21), language='python')
            _note(
                '<b>5 lags</b>: lag_1/2/4/8/52 — lag_52 is the same-week-last-year anchor (year-over-year seasonality).<br>'
                '<b>4 rolling</b>: mean at 4/8/13 weeks + 4-week std.<br>'
                '<b>3 calendar</b>: week_of_year, month, year.<br>'
                '<b>3 store context</b>: snap, has_event, event_type_enc.<br>'
                '<b>2 price</b>: sell_price, price_change_pct.<br>'
                '<b>2 item priors</b>: dept_mean_sales, cat_mean_sales.',
                border='#1565c0',
            )

        with col2:
            st.markdown('#### Leakage controls (critical)')
            st.code(
                _excerpt(features_src, 47, 62),
                language='python',
            )
            _note(
                '<b>Lag features</b>: <code>shift(n)</code> per SKU — lag_1 at week t = sales[t-1].<br>'
                '<b>Rolling features</b>: <code>shift(1).rolling(w)</code> — excludes the current week before windowing.<br>'
                '<b>Price</b>: <code>ffill</code> within item only — no <code>bfill</code>. '
                'Pre-launch NaN rows stay NaN and are dropped by <code>dropna(FEATURE_COLS)</code> at train time. '
                'bfill would pull a future price backward into 87k pre-launch rows across 60% of SKUs.',
                border='#e53935',
            )

        st.markdown('#### Full compute_features()')
        st.code(features_src, language='python')

    with tab2:
        _file_badge('build_features.py')
        build_src = _read('build_features.py')
        st.markdown(
            '**One-time precompute.** Reads all 847k raw rows, runs `compute_features()`, '
            'clears the `features` table, and writes the result back. Safe to re-run at any time — '
            'always clears first, never accumulates stale rows.'
        )
        _note(
            '⚠️ <b>Critical invariant</b>: rebuild the feature store whenever <code>features.py</code> changes. '
            'If you edit a feature and forget to rebuild, the backtest silently trains on stale data. '
            'The smoke test\'s staleness check catches this before a full run.',
            border='#e53935',
        )
        st.code(build_src, language='python')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ML PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '🤖  ML Pipeline':
    _header('🤖 ML Pipeline', 'train.py · forecast.py · evaluate.py', COLORS[section])

    train_src    = _read('xai_forecast/train.py')
    forecast_src = _read('xai_forecast/forecast.py')
    evaluate_src = _read('xai_forecast/evaluate.py')

    tab1, tab2, tab3 = st.tabs(['train.py', 'forecast.py', 'evaluate.py'])

    with tab1:
        _file_badge('xai_forecast/train.py')
        st.markdown('**One global LightGBM model across all SKUs.** No per-SKU models — the model learns from cross-SKU patterns.')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.code(train_src, language='python')
        with col2:
            st.markdown('#### Why Tweedie?')
            _note(
                '<b>objective: tweedie</b> — correct for zero-heavy intermittent demand data (64% zero-sale weeks).<br><br>'
                '<b>variance_power=1.5</b> — between Poisson (1.0) and Gamma (2.0). Appropriate for sparse count data.<br><br>'
                '<b>log-link</b> — Tweedie uses a log-link internally. This means SHAP values are in log-margin space: '
                '<code>base_value_log + Σ(shap) = log(prediction)</code>. Rankings by |shap| are valid; '
                'raw shap values need exponentiation to convert back to unit space.',
                border='#2e7d32',
            )
            st.markdown('#### Training data')
            _note(
                '<code>dropna(subset=FEATURE_COLS + ["y"])</code> — pre-launch SKUs (all-NaN lags) are excluded from training. '
                'They can still receive forecasts (via fillna(0) in make_forecasts), '
                'but they don\'t corrupt the training set.',
                border='#2e7d32',
            )

    with tab2:
        _file_badge('xai_forecast/forecast.py')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.code(forecast_src, language='python')
        with col2:
            st.markdown('#### Pre-launch SKU handling')
            _note(
                '<code>fillna(0)</code> before predict — a SKU in its pre-launch weeks has all-NaN lag features. '
                'Imputing to 0 lets the model produce a (garbage) forecast rather than crashing. '
                'If that SKU had actual sales > 0 in that week, it will be scored against a garbage forecast '
                'and may trigger a bad-week flag. <code>backtest.py</code> counts and logs these rows.',
                border='#2e7d32',
            )
            _note(
                '<code>clip(min=0)</code> — Tweedie predictions are non-negative by construction, '
                'but floating-point arithmetic can produce tiny negatives. Clip ensures h1 ≥ 0.',
                border='#2e7d32',
            )

    with tab3:
        _file_badge('xai_forecast/evaluate.py')
        st.code(evaluate_src, language='python')

        st.markdown('#### Bad-week detection logic')
        col1, col2 = st.columns(2)
        with col1:
            _note(
                '<b>WMAPE</b> = Σ|error| / Σactual — volume-weighted, not dominated by near-zero-actual SKUs. '
                'A 100% MAPE on a SKU selling 1 unit is noise. WMAPE weights by actual volume.',
                border='#2e7d32',
            )
        with col2:
            _note(
                '<b>shift(1) on baseline</b> — critical. Without it, the current week\'s own WMAPE enters its rolling mean, '
                'inflating the denominator of its z-score. With shift(1), each week is scored against the prior 8 weeks only. '
                'A spike is loud against the stable prior period.',
                border='#e53935',
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — XAI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '🔍  XAI Engine':
    _header('🔍 XAI Engine', 'xai.py — SHAP · Counterfactual · Contrastive', COLORS[section])

    xai_src = _read('xai_forecast/xai.py')

    st.markdown(
        'For each bad week, **all valid non-pre-launch SKUs** receive three types of explanation. '
        'All are computed using the **exact retrain checkpoint** that produced that week\'s forecast.'
    )

    tab1, tab2, tab3 = st.tabs(['SHAP', 'Counterfactual', 'Contrastive'])

    with tab1:
        st.markdown('#### What it answers: *"Which features pushed the forecast up or down?"*')
        col1, col2 = st.columns([3, 2])
        with col1:
            _file_badge('xai_forecast/xai.py — shap_payloads()')
            st.code(_excerpt(xai_src, 32, 94), language='python')
        with col2:
            st.markdown('#### Log-margin space')
            _note(
                'Tweedie uses a log-link. SHAP values are additive in log-margin space:<br><br>'
                '<code>base_value_log + Σ(top5 shap) + other_features_shap ≈ log(prediction)</code><br><br>'
                'The <code>other_features_shap</code> term is the sum of the 21 non-top-5 features — '
                'included so the waterfall chart in the dashboard reconciles exactly to the actual prediction.',
                border='#e65100',
            )
            st.markdown('#### signed_error / direction')
            _note(
                'Each SHAP row includes:<br>'
                '<code>signed_error</code>: (pred − actual) / actual × 100 (+ = over-forecast)<br>'
                '<code>direction</code>: "over" or "under"<br><br>'
                'Guard: <code>if actual > 0</code> — not <code>>= 0</code>. '
                'Zero-actual rows are already excluded by evaluate_h1, but the guard is explicit.',
                border='#e65100',
            )
            st.markdown('#### Return value: (rows, shap_cache)')
            _note(
                '<code>shap_cache</code>: uid → 1-D SHAP array. '
                'Passed to <code>contrastive_payloads</code> as <code>bad_shap_cache</code> '
                'so the bad-item SHAP is not recomputed a second time.',
                border='#e65100',
            )

    with tab2:
        st.markdown('#### What it answers: *"What if SNAP / event / price change hadn\'t happened?"*')
        col1, col2 = st.columns([3, 2])
        with col1:
            _file_badge('xai_forecast/xai.py — counterfactual_payloads()')
            st.code(_excerpt(xai_src, 21, 25), language='python')
            st.code(_excerpt(xai_src, 97, 145), language='python')
        with col2:
            st.markdown('#### Three scenarios')
            for s, desc in [
                ('no_snap', 'Set snap=0 — removes SNAP promotion effect'),
                ('no_event', 'Set has_event=0, event_type_enc=0 — removes calendar event'),
                ('no_price_change', 'Set price_change_pct=0 — removes pricing shock'),
            ]:
                _note(f'<b>{s}</b>: {desc}', border='#e65100')

            st.markdown('#### was_active flag')
            _note(
                'Before we zero a feature, we check if it was non-zero for this SKU this week. '
                'If snap=0 already, zeroing it is a no-op — delta_pct ≈ 0, misleading. '
                '<code>was_active=False</code> lets the dashboard gray out inactive scenarios '
                'rather than showing them as meaningful counterfactuals.',
                border='#e65100',
            )

    with tab3:
        st.markdown('#### What it answers: *"Compared to a week the model got right — what was different?"*')
        col1, col2 = st.columns([3, 2])
        with col1:
            _file_badge('xai_forecast/xai.py — contrastive_payloads()')
            st.code(_excerpt(xai_src, 148, 271), language='python')
        with col2:
            st.markdown('#### Same-WOY constraint')
            _note(
                'A reference week must have the same ISO week-of-year as the bad week. '
                'This controls for seasonality: comparing week 1 of 2015 to week 27 of 2014 '
                'would conflate winter vs summer patterns with structural model differences. '
                'If no same-WOY good week exists — skip. No fallback.',
                border='#e65100',
            )
            st.markdown('#### Batching by ref week')
            _note(
                'Multiple items can share the same reference week (e.g. all WOY-1 items comparing to 2014-01-04). '
                'Reference features are loaded once per unique ref week, not once per item. '
                'SHAP on ref items is also computed in a single batch per ref week.',
                border='#e65100',
            )
            st.markdown('#### SHAP diff')
            _note(
                '<code>shap_diff = bad_shap − good_shap</code> for each feature. '
                'Sorted by <code>|shap_diff|</code> descending. '
                'A large positive diff on lag_1 means lag_1 pushed the prediction up much more in the bad week '
                'than in the good reference — a structural divergence in that feature\'s signal.',
                border='#e65100',
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INSIGHTS MODULE
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '💬  Insights Module':
    _header(
        '💬 Insights Module',
        'xai_forecast/insights/ — detectors → planner → hypothesis → critic → synthesis',
        COLORS[section],
    )

    graph_src     = _read('xai_forecast/insights/graph.py')
    detectors_src = _read('xai_forecast/insights/detectors.py')
    agents_src    = _read('xai_forecast/insights/agents.py')
    tools_src     = _read('xai_forecast/insights/tools.py')

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        'Architecture',
        'Detectors',
        'Planner + tools',
        'Critic + grounding',
        'Prompt constants',
    ])

    with tab1:
        st.markdown('#### Evidence-first design philosophy')
        _note(
            '<b>Deterministic detectors fire first</b> — they surface candidate findings from real data thresholds. '
            'The LLM never sees raw DB rows directly; it only sees the bounded evidence pack each detector assembled.<br><br>'
            '<b>LLM interprets, not discovers</b> — Flash writes hypotheses, Pro critiques them. '
            'The critic is the single quality gate. A deterministic grounding advisory is forwarded to the critic '
            'as context but does not gate or mutate confidence.',
            border='#6a1b9a',
        )

        st.markdown('#### LangGraph flow')
        st.markdown(
            '''
<div style="background:#f3e5f5;padding:16px 20px;border-radius:8px;font-size:0.85rem;line-height:2">
<b>detect_candidates</b> (deterministic, no LLM)<br>
&nbsp;&nbsp;→ run_all_detectors(conn) → list[CandidateFinding]<br>
<br>
<b>fan-out via Send()</b> — one review_finding node per candidate<br>
<br>
<b>review_finding</b> (per-finding):<br>
&nbsp;&nbsp;1. run_planner(Flash) → choose 1-4 read-tools<br>
&nbsp;&nbsp;2. _enrich_evidence(conn, tools) → merge read-tool results into evidence dict<br>
&nbsp;&nbsp;3. run_hypothesis(Flash) → grounded hypothesis with evidence_refs<br>
&nbsp;&nbsp;4. grounding advisory → check refs against enriched keys; forward to critic<br>
&nbsp;&nbsp;5. run_critic(Pro) → accepted / rejected / needs_review<br>
<br>
<b>fan-in: ledger_rows (Annotated[list, add] reducer)</b><br>
<br>
<b>synthesize</b>:<br>
&nbsp;&nbsp;→ filter accepted findings → run_business_synthesis + run_technical_synthesis (concurrent asyncio.gather) → DS view + business view<br>
&nbsp;&nbsp;→ stored in insight_summary table<br>
</div>
''',
            unsafe_allow_html=True,
        )

        st.markdown('#### StateGraph + closure factory')
        _note(
            '<code>_State(TypedDict)</code> — three keys: <code>candidates</code>, '
            '<code>ledger_rows: Annotated[list, add]</code>, <code>summary</code>.<br><br>'
            '<b>Fan-in reducer</b>: <code>Annotated[list, add]</code> on <code>ledger_rows</code> '
            'means each <code>review_finding</code> node appends its <code>LedgerRow</code> rather than '
            'overwriting — no manual state merging needed.<br><br>'
            '<b>Closure factory</b>: <code>_build_graph(conn, client)</code> captures <code>conn</code> and '
            '<code>client</code> in node-function closures — they never enter the LangGraph state dict. '
            'This avoids <code>KeyError</code> when LangGraph passes only the node-output delta to edge routers, '
            'and keeps state serialisation-safe.',
            border='#6a1b9a',
        )
        _file_badge('xai_forecast/insights/graph.py — _State + _build_graph')
        st.code(_excerpt(graph_src, 108, 140), language='python')

    with tab2:
        st.markdown('#### Six deterministic detectors — fire before any LLM call')
        _note(
            'Each detector reads from SQLite directly (no LLM). '
            'Fires only when evidence meets a real-data threshold. '
            'Returns a <code>CandidateFinding</code> (finding_type, score 0-1, summary, evidence dict) '
            'or <code>None</code> if the threshold is not met.',
            border='#6a1b9a',
        )

        detectors_table = {
            'over_forecast_bias': ('70% of SHAP payloads must be over-forecasts', 'Systematic bias: risk is over-ordering, not stockouts'),
            'dominant_driver': ('One feature in >60% of SHAP payloads', 'Model over-anchors on one feature across bad weeks'),
            'demand_cliff': ('lag_1 ≥ 3× actual for ≥3 items', 'Momentum over-anchoring: sales dropped after model anchored on high lag_1'),
            'external_coincidence': ('Bad week + heat wave / gas spike / sentiment crisis', 'Correlation with external conditions (stated as correlation only)'),
            'counterfactual_material': ('Zeroing SNAP/event/price moves prediction >5%', 'Feature was causally active and quantifiably affected the forecast'),
            'contrastive_gap': ('Always fires if contrastive data exists', 'Structural SHAP diff vs same-WOY good reference week'),
        }
        for det, (threshold, interpretation) in detectors_table.items():
            with st.expander(f'**{det}** — threshold: {threshold}'):
                st.markdown(f'**Interpretation:** {interpretation}')

        st.markdown('#### Example — detect_over_forecast_bias')
        _file_badge('xai_forecast/insights/detectors.py')
        st.code(_excerpt(detectors_src, 44, 80), language='python')

        st.markdown('#### run_all_detectors — sorted by score descending')
        st.code(_excerpt(detectors_src, 390, 409), language='python')

    with tab3:
        st.markdown('#### Planner — Flash decides which read-tools to call')
        _note(
            'Before writing a hypothesis, Flash receives the finding type, score, summary, and '
            'available evidence keys, then returns a JSON list of 1–4 read-tools to call. '
            'This avoids calling every tool for every finding — dominant_driver gets '
            '<code>read_recurring_drivers</code> + <code>read_model_metadata</code>; '
            'external_coincidence always gets <code>read_external_signals</code>.',
            border='#6a1b9a',
        )
        _file_badge('xai_forecast/insights/agents.py — run_planner()')
        st.code(_excerpt(agents_src, 154, 176), language='python')

        st.markdown('#### Seven read-tools (no LLM, no side effects)')
        tools_list = [
            ('read_forecast_accuracy', 'Global MAPE stats, bad/good week rates, worst week'),
            ('read_bad_weeks', 'All flagged bad weeks with z-scores and avg MAPE'),
            ('read_xai_findings', 'SHAP/CF/contrastive payloads — sample from worst week'),
            ('read_demand_trajectory', 'Actual sales + lag_1 + rolling mean + forecast for one SKU over time'),
            ('read_external_signals', 'LA weather, CA gas price, consumer sentiment for a specific week'),
            ('read_model_metadata', 'Model config + global feature importance from last checkpoint'),
            ('read_recurring_drivers', 'Feature appearance frequency across all bad-week SHAP payloads'),
        ]
        for tool, desc in tools_list:
            _note(f'<code>{tool}</code>: {desc}', border='#6a1b9a')

        st.markdown('#### _enrich_evidence — calls chosen tools and merges results')
        _file_badge('xai_forecast/insights/graph.py — _enrich_evidence()')
        st.code(_excerpt(graph_src, 40, 92), language='python')

    with tab4:
        st.markdown('#### Two-model split: Flash for volume, Pro for quality gate')
        col1, col2 = st.columns(2)
        with col1:
            _note(
                '<b>Flash (deepseek-v4-flash)</b>:<br>'
                '• run_planner — choose read-tools<br>'
                '• run_hypothesis — interpret evidence<br>'
                '• run_synthesis — combine accepted findings<br>'
                'Temperature 0.2 (planner/hypothesis) or 0.0 (synthesis). MAX_TOKENS_FLASH = 3000.',
                border='#6a1b9a',
            )
        with col2:
            _note(
                '<b>Pro (deepseek-v4-pro)</b>:<br>'
                '• run_critic — single quality gate<br>'
                'Rejects overclaim, forbids causal external claims,<br>'
                'downgrades weak evidence, sets final confidence.<br>'
                'Temperature 0.0 (deterministic governance artifact). MAX_TOKENS_PRO = 4096.',
                border='#6a1b9a',
            )

        st.markdown('#### Grounding check — advisory, not a gate')
        _note(
            'After hypothesis, <code>_build_evidence_key_set(enriched)</code> builds a set containing '
            'both full dot-paths (<code>model_metadata.global_feature_importance</code>) AND bare leaf names '
            '(<code>global_feature_importance</code>). Flash\'s <code>evidence_refs</code> are matched '
            'tolerantly: <code>_normalize_ref</code> converts <code>[0]</code> bracket notation to <code>.0</code> '
            'before comparison.<br><br>'
            '<b>Key design decision</b>: grounding result goes into <code>grounding_advisory: '
            '{grounding_ok, missing_refs}</code> forwarded to the Pro critic as payload context — '
            'it does NOT mutate <code>hypothesis.confidence</code> and does NOT bypass the critic. '
            'The critic is the single quality gate.',
            border='#e53935',
        )
        _file_badge('xai_forecast/insights/graph.py — review_finding (grounding + critic)')
        st.code(_excerpt(graph_src, 139, 186), language='python')

        st.markdown('#### run_critic — Pro quality gate')
        _file_badge('xai_forecast/insights/agents.py — run_critic()')
        st.code(_excerpt(agents_src, 210, 260), language='python')

    with tab5:
        st.markdown('#### Five prompt constants — run `/prompt-audit` before editing any')
        _note(
            '⚠️ All five are <code>*_PROMPT</code> constants in <code>agents.py</code>. '
            'Per project rules, run <code>/prompt-audit</code> before committing any edit.',
            border='#e53935',
        )

        st.markdown('#### PLANNER_PROMPT — Flash: choose read-tools')
        _file_badge('xai_forecast/insights/agents.py — PLANNER_PROMPT (line 31)')
        st.code(_excerpt(agents_src, 31, 58), language='python')

        st.markdown('#### HYPOTHESIS_PROMPT — Flash: write grounded hypothesis')
        _file_badge('xai_forecast/insights/agents.py — HYPOTHESIS_PROMPT (line 60)')
        _note(
            '<b>CRITICAL section</b>: three banned failure modes — '
            '(1) model-feature causation (SHAP shows model weights, not demand drivers), '
            '(2) counterfactual extrapolation (sensitivity tests are not production causal claims), '
            '(3) coverage editorializing (state the %, never infer reliability consequences). '
            'The CRITIC_PROMPT enforces the same three rules as explicit reject triggers.',
            border='#e53935',
        )
        st.code(_excerpt(agents_src, 60, 125), language='python')

        st.markdown('#### CRITIC_PROMPT — Pro: reject overclaim (temperature=0)')
        _file_badge('xai_forecast/insights/agents.py — CRITIC_PROMPT (line 126)')
        st.code(_excerpt(agents_src, 126, 164), language='python')

        st.markdown('#### BUSINESS_SYNTHESIS_PROMPT — Flash: VP-facing brief (temperature=0)')
        _file_badge('xai_forecast/insights/agents.py — BUSINESS_SYNTHESIS_PROMPT (line 166)')
        _note(
            'Zero jargon. Renders: headline, progress (health_verdict, diagnosis), '
            'plan (phases, impact), limitations, risk_direction, overall_confidence.',
            border='#6a1b9a',
        )
        st.code(_excerpt(agents_src, 166, 201), language='python')

        st.markdown('#### TECHNICAL_SYNTHESIS_PROMPT — Flash: DS-facing levers (temperature=0)')
        _file_badge('xai_forecast/insights/agents.py — TECHNICAL_SYNTHESIS_PROMPT (line 203)')
        _note(
            'Buckets: feature_engineering | model_param | workflow | algorithm. '
            'Each lever must cite a specific statistic from the accepted findings.',
            border='#6a1b9a',
        )
        st.code(_excerpt(agents_src, 203, 250), language='python')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '⚙️  Orchestration':
    _header('⚙️ Orchestration', 'backtest.py · smoke_test.py · data_quality.py', COLORS[section])

    backtest_src = _read('backtest.py')
    smoke_src    = _read('smoke_test.py')

    tab1, tab2 = st.tabs(['backtest.py — main loop', 'smoke_test.py — sanity checks'])

    with tab1:
        _file_badge('backtest.py')
        st.markdown('**Sliding-window backtest.** Trains a new LightGBM every 4 weeks over ~120 weeks.')

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown('#### Constants + setup')
            st.code(_excerpt(backtest_src, 41, 80), language='python')
            st.markdown('#### Main loop — train → forecast → evaluate')
            st.code(_excerpt(backtest_src, 82, 113), language='python')
        with col2:
            st.markdown('#### Per-checkpoint XAI')
            _note(
                '<code>all_models</code>: dict mapping retrain_cutoff → LGBMRegressor.<br>'
                '<code>week_to_cutoff</code>: maps each forecast_week to its retrain_cutoff.<br><br>'
                'This means XAI for a 2013 bad week uses the 2013 checkpoint, not the 2016 final model. '
                'SHAP is faithful to what the model knew at the time.',
                border='#37474f',
            )
            st.markdown('#### Clean slate at start')
            _note(
                'Every run begins with <code>DELETE FROM forecasts; DELETE FROM evaluations;</code>. '
                'xai_results and insight tables are managed by their own stages '
                '(<code>run_xai.py</code>, <code>generate_insights.py</code>) — each clears its own tables. '
                'No partial runs, no orphan rows. INSERT OR REPLACE provides additional idempotency within a run.',
                border='#37474f',
            )
            st.markdown('#### Last two weeks excluded')
            _note(
                '<code>backtest_weeks = weeks[TRAIN_WINDOW:-2]</code> — M5 evaluation file ends 2 days '
                'into the final fiscal week (partial week, artificially low sales). '
                'Forecasting it produces a spurious ~215% MAPE spike. Excluded.',
                border='#e53935',
            )

        st.markdown('#### Evaluations + flagging (after main loop)')
        _note(
            '<code>flag_bad_weeks</code> runs on the full concatenated eval dataframe — needs all weeks to compute '
            'the rolling WMAPE z-score baseline. Then <code>insert_evaluations</code> writes everything in one batch.',
            border='#37474f',
        )
        st.code(_excerpt(backtest_src, 108, 142), language='python')

    with tab2:
        _file_badge('smoke_test.py')
        st.markdown(
            '**Pre-flight check** before a full backtest run (~50s). '
            'Writes to `db/smoke.db` — never contaminates `db/forecasting.db`.'
        )
        _note(
            '<b>Checks performed:</b><br>'
            '1. <b>Feature staleness</b>: recompute features for 10 SKUs × 3 weeks, diff against feature store. Fails if any mismatch.<br>'
            '2. <b>Parallel forecast</b>: runs 10 weeks concurrently (ThreadPoolExecutor), validates h1≥0, SHAP additivity, JSON roundtrip.<br>'
            '3. <b>Contrastive</b>: validates same-WOY selection and shap_diff math.<br>'
            '4. <b>SHAP additivity</b>: base_value_log + Σ(top5) + other_features_shap ≈ log(prediction) ± 0.01.<br>'
            '5. <b>DeepSeek API probe</b>: one live Flash call. Fails loudly on bad config or missing key.',
            border='#37474f',
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '🗄️  Storage':
    _header('🗄️ Storage', 'db.py · 7 migrations · SQLite WAL', COLORS[section])

    db_src = _read('xai_forecast/db.py')

    tab1, tab2, tab3 = st.tabs(['db.py', 'Schema & migrations', 'INSERT OR REPLACE pattern'])

    with tab1:
        _file_badge('xai_forecast/db.py')
        col1, col2 = st.columns([3, 2])
        with col1:
            st.code(db_src, language='python')
        with col2:
            st.markdown('#### get_conn — auto-schema')
            _note(
                '<code>get_conn(path)</code> applies all <code>migrations/*.sql</code> on every connection. '
                'No manual migration step. SQL files use <code>CREATE TABLE IF NOT EXISTS</code> '
                'so re-applying is a no-op. To add schema: add <code>00N_description.sql</code>. '
                'Never edit existing migration files.',
                border='#00695c',
            )
            st.markdown('#### WAL mode')
            _note(
                'WAL (Write-Ahead Logging) allows concurrent reads while a write is in progress. '
                'Important during the backtest — Streamlit dashboard can be open while backtest writes.',
                border='#00695c',
            )
            st.markdown('#### insight tables')
            _note(
                '<code>insight_findings</code>: one row per candidate finding — '
                'finding_id, finding_type, status (accepted/rejected/needs_review), confidence, '
                'evidence JSON, hypothesis JSON, critic_notes. PRIMARY KEY (finding_id).<br><br>'
                '<code>insight_summary</code>: single row keyed on "overall" — '
                'data_scientist JSON, business_leader JSON, model_flash, model_critic, created_at. '
                'Written by <code>generate_insights.py</code> (mandatory LLM — fails loudly without key).',
                border='#00695c',
            )

    with tab2:
        st.markdown('#### 7 migration files (applied in sorted order)')
        for mig_file in sorted((ROOT / 'migrations').glob('*.sql')):
            with st.expander(mig_file.name):
                st.code(mig_file.read_text(encoding='utf-8'), language='sql')

        st.markdown('#### Table ownership')
        st.dataframe(
            {
                'Table': [
                    'weekly_sales', 'calendar', 'prices', 'item_meta',
                    'features', 'forecasts', 'evaluations', 'xai_results',
                    'external_signals', 'insight_findings', 'insight_summary',
                ],
                'Written by': [
                    'ingest.py', 'ingest.py', 'ingest.py', 'ingest.py',
                    'build_features.py', 'backtest.py', 'backtest.py', 'run_xai.py',
                    'ingest_external.py', 'generate_insights.py', 'generate_insights.py',
                ],
                'Purpose': [
                    'Raw weekly unit sales per SKU',
                    'SNAP, event flags per week',
                    'Weekly avg sell price per SKU',
                    'dept_id, cat_id, mean sales priors',
                    'Precomputed feature matrix (847k rows)',
                    'h=1 predictions per SKU per week',
                    'MAPE, MAE, WMAPE z-score, bad-week flag',
                    'JSON payloads: shap / counterfactual / contrastive',
                    'Per-week: LA weather, CA gas price, consumer sentiment',
                    'Per-finding: status, confidence, evidence JSON, hypothesis JSON, critic notes',
                    'Single overall row: DS view + business view + model names',
                ],
            },
            use_container_width=True,
        )

    with tab3:
        st.markdown('#### Why INSERT OR REPLACE everywhere?')
        _note(
            'All output tables use <code>INSERT OR REPLACE</code> keyed on their primary keys. '
            'Each stage clears its own tables at the start (clean-slate DELETE), '
            'then INSERT OR REPLACE provides a belt-and-suspenders safety net for partial writes.<br><br>'
            'This makes every stage independently idempotent: re-running '
            '<code>generate_insights.py</code> alone always produces a clean result without '
            'touching forecasts, evaluations, or xai_results.',
            border='#00695c',
        )
        st.markdown('#### Example: insert_forecasts (backtest)')
        st.code(_excerpt(db_src, 113, 119), language='python')
        st.markdown('#### Example: insert_insight_finding (insights module)')
        st.code(_excerpt(db_src, 164, 172), language='python')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '📊  Dashboard':
    _header('📊 Dashboard', 'app.py — single-page insights dashboard', COLORS[section])

    app_src = _read('app.py')

    st.markdown('**Run:** `uv run streamlit run app.py` → `localhost:8501`')

    st.markdown('#### Single-page layout (top → bottom)')
    sections_info = [
        ('1. MAPE time series', 'lines 96–132',
         'Weekly avg MAPE line chart + bad-week markers (×). Four KPI tiles: total weeks, bad weeks, overall avg MAPE, worst week MAPE.'),
        ('2. Insights summary', 'lines 135–195',
         'Two-column layout: Data Scientist view (headline, summary, top issues, recommended actions) + Business Leader view (headline, summary, risk direction, limitations, improvement plan). Reads from insight_summary table.'),
        ('3. Findings ledger', 'lines 198–265',
         'Dataframe showing all findings (status, confidence, critic notes preview). Selectbox to inspect any finding — shows DS/business explanations + raw evidence JSON expander.'),
        ('4. XAI drill-down', 'lines 268–434',
         'Pick a bad week + item → SHAP waterfall (Plotly), Counterfactual bar chart (active scenarios only), Contrastive grouped bar + diff table.'),
    ]
    for title, loc, desc in sections_info:
        with st.expander(f'**{title}** — {loc}'):
            st.markdown(desc)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        'Data loaders', 'MAPE + Insights', 'Findings ledger', 'SHAP waterfall', 'CF + Contrastive',
    ])

    with tab1:
        st.markdown('#### Caching strategy')
        st.code(_excerpt(app_src, 42, 69), language='python')
        _note(
            '<code>@st.cache_data</code> on all stable DB reads: '
            '<code>_week_summary</code>, <code>_evaluations</code>, '
            '<code>_insight_summary</code>, <code>_insight_findings</code>.<br><br>'
            '<code>_xai(week, item)</code> is NOT cached — called on demand per item selection, '
            'low volume (one week × one item at a time). '
            'Cache is cleared automatically when Streamlit reruns.',
            border='#0277bd',
        )

    with tab2:
        st.markdown('#### MAPE chart + KPI tiles')
        st.code(_excerpt(app_src, 100, 132), language='python')
        st.markdown('#### Two-perspective insights summary')
        st.code(_excerpt(app_src, 135, 195), language='python')
        _note(
            'If <code>insight_summary</code> is empty (insights not yet generated), '
            'shows an info banner with the command to run. '
            'Risk direction is color-coded: over-stock=red, under-stock=orange, mixed=gray.',
            border='#0277bd',
        )

    with tab3:
        st.markdown('#### Findings ledger — auditable finding-by-finding view')
        st.code(_excerpt(app_src, 198, 265), language='python')
        _note(
            'Each row in the ledger corresponds to one candidate finding that went through the '
            'full detector → planner → hypothesis → critic chain. '
            'Status badge (accepted/rejected/needs_review) + confidence badge come from the Pro critic. '
            'Raw evidence JSON is shown in an expander for full traceability.',
            border='#0277bd',
        )

    with tab4:
        st.markdown('#### SHAP waterfall — top 5 features + residual')
        st.code(_excerpt(app_src, 303, 345), language='python')
        _note(
            'Plotly Waterfall with <code>base=base_log</code>. The "other N features" bar is the residual '
            '(sum of non-top-5 SHAP values) — ensures the waterfall ends exactly at log(prediction). '
            '<code>n_other = len(FEATURE_COLS) - len(feats)</code> (uses FEATURE_COLS constant, not magic 19).',
            border='#0277bd',
        )

    with tab5:
        st.markdown('#### Counterfactual — active scenarios bar chart')
        st.code(_excerpt(app_src, 347, 391), language='python')
        _note(
            'Only active scenarios are plotted as bars. Inactive scenarios (was_active=False) '
            'appear in the full dataframe below with "No (inactive)" label — visible but not meaningful.',
            border='#0277bd',
        )
        st.markdown('#### Contrastive — grouped bar: bad vs good reference week')
        st.code(_excerpt(app_src, 393, 434), language='python')
        _note(
            'Two bar series (red = bad week, blue = good reference) grouped by feature. '
            'The diff table shows <code>shap_diff = bad_shap − good_shap</code>. '
            'If no same-WOY good week exists for this SKU, the tab shows a coverage note '
            '(59% of items have no qualifying reference week).',
            border='#0277bd',
        )

    st.markdown('---')
    st.markdown('#### Full app.py')
    with st.expander('Show full source'):
        st.code(app_src, language='python')
