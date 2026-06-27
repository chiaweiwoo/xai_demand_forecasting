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
    '💬  LLM Narrative',
    '⚙️  Orchestration',
    '🗄️  Storage',
    '📊  Dashboard',
]
section = st.sidebar.radio('Navigate', SECTIONS, label_visibility='collapsed')

COLORS = {
    '🗺️  Overview':      '#1a73e8',
    '📦  Data Layer':    '#1565c0',
    '🤖  ML Pipeline':   '#2e7d32',
    '🔍  XAI Engine':    '#e65100',
    '💬  LLM Narrative': '#6a1b9a',
    '⚙️  Orchestration': '#37474f',
    '🗄️  Storage':       '#00695c',
    '📊  Dashboard':     '#0277bd',
}

st.sidebar.markdown('---')
st.sidebar.markdown(
    '**Layer map**\n'
    '- 📦 Data → features table\n'
    '- 🤖 ML → forecasts + bad-week flags\n'
    '- 🔍 XAI → SHAP / CF / contrastive\n'
    '- 💬 LLM → plain-English narratives\n'
    '- 📊 Dashboard → leader view'
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
        'for the top-50 worst SKUs, then synthesises them into a plain-English narrative a business '
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
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>build_features.py</b> → SQLite: <i>features</i> (847k rows, 19 cols, ~46s one-time)<br>

<br><b style="color:#2e7d32">ML LAYER</b><br>
&nbsp;&nbsp;<b>backtest.py</b> ← feature store (SQL SELECT per week, ~2s)<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>train.py</b>: LightGBM Tweedie (retrain every 4 weeks, 3-year window, ×30 checkpoints)<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>forecast.py</b>: h=1 predictions per SKU<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>evaluate.py</b>: WMAPE z-score ≥ 1.5 → bad-week flag<br>

<br><b style="color:#e65100">XAI LAYER</b><br>
&nbsp;&nbsp;For each bad week (top 50 worst SKUs):<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>xai.py / SHAP</b>: TreeSHAP — top-5 drivers in log-margin space<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>xai.py / Counterfactual</b>: zero out SNAP/event/price → measure prediction delta<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ <b>xai.py / Contrastive</b>: diff SHAP vs a good same-WOY reference week<br>

<br><b style="color:#6a1b9a">LLM LAYER</b><br>
&nbsp;&nbsp;<b>narrate.py</b>: dossier builders (pure) → DeepSeek V4 Flash (OpenAI SDK)<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ Week narrative · Item narrative · Executive synthesis<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ Grounding check: primary_driver must exist in evidence features<br>

<br><b style="color:#00695c">STORAGE</b><br>
&nbsp;&nbsp;<b>db.py</b>: SQLite (WAL mode) — 9 tables, schema auto-applied via 4 migrations<br>
&nbsp;&nbsp;All writes are INSERT OR REPLACE — backtest is idempotent, re-runnable<br>

<br><b style="color:#0277bd">DASHBOARD</b><br>
&nbsp;&nbsp;<b>app.py</b>: Streamlit — 4 pages: Overview · Bad Week Drilldown · Recurring Drivers · XAI Explorer<br>
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
    c5.metric('Features', '19', '5 lag + 4 rolling + …')

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
        st.markdown('**Single source of truth** for all 19 features. Used by `build_features.py`, `backtest.py`, `smoke_test.py`, and all tests.')

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('#### Feature columns (19 total)')
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
        'For each bad week, the **top 50 worst SKUs** receive three types of explanation. '
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
                'The <code>other_features_shap</code> term is the sum of the 14 non-top-5 features — '
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
# SECTION 5 — LLM NARRATIVE
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '💬  LLM Narrative':
    _header('💬 LLM Narrative', 'narrate.py — DeepSeek V4 Flash + grounding check', COLORS[section])

    narrate_src = _read('xai_forecast/narrate.py')

    tab1, tab2, tab3, tab4 = st.tabs([
        'Architecture',
        'Dossier builders',
        'generate() + grounding',
        'Prompt constants',
    ])

    with tab1:
        st.markdown('#### How the three narrative types relate')
        st.markdown(
            '''
<div style="background:#f3e5f5;padding:16px 20px;border-radius:8px;font-size:0.85rem;line-height:2">
<b>Week narrative</b> (scope=week, key=forecast_week)<br>
&nbsp;&nbsp;→ build_week_dossier: aggregated SHAP across top-50 SKUs<br>
&nbsp;&nbsp;→ WEEK_NARRATIVE_PROMPT → "Recent sales trend shifted unexpectedly this week…"<br>
<br>
<b>Item narrative</b> (scope=item, key=week::item_id)<br>
&nbsp;&nbsp;→ build_item_dossier: SHAP + CF + contrastive for one SKU<br>
&nbsp;&nbsp;→ ITEM_NARRATIVE_PROMPT → "This product was over-forecast by 40%…"<br>
<br>
<b>Executive narrative</b> (scope=executive, key=overall)<br>
&nbsp;&nbsp;→ build_executive_dossier: recurring driver frequencies across ALL bad weeks<br>
&nbsp;&nbsp;→ EXECUTIVE_NARRATIVE_PROMPT → "Across 15 bad weeks, lag features dominated failures…"<br>
</div>
''',
            unsafe_allow_html=True,
        )

        st.markdown('#### Dossier → LLM flow')
        _note(
            '1. <b>Dossier builder</b> (pure function, no network) — assembles evidence JSON from DB rows.<br>'
            '2. <b>generate()</b> — sends (system_prompt, dossier) to DeepSeek, gets JSON back.<br>'
            '3. <b>Grounding check</b> — validates primary_driver is in the evidence feature list.<br>'
            '4. <b>Store</b> — insert_narrative(scope, key, payload) → narratives table.',
            border='#6a1b9a',
        )

    with tab2:
        st.markdown('#### build_week_dossier — aggregates SHAP across all items in a bad week')
        st.code(_excerpt(narrate_src, 100, 141), language='python')

        st.markdown('#### compute_recurring_drivers — cross-week frequency analysis')
        _note(
            'Single source of truth used by both <code>backtest.py</code> (executive narrative) and '
            '<code>app.py</code> (Recurring Drivers page).<br><br>'
            'Returns per-feature: <code>count</code>, <code>pct_payloads</code> (% of all SHAP explanations), '
            '<code>n_weeks</code>, <code>pct_bad_weeks</code> (% of distinct bad weeks).<br><br>'
            '<b>pct_payloads vs pct_bad_weeks</b>: different denominators. '
            'pct_payloads counts SKU explanations (a feature at 80% appeared in 80% of top-50 slots). '
            'pct_bad_weeks counts distinct weeks (80% = feature appeared in 80% of all flagged weeks). '
            'The executive prompt uses pct_bad_weeks for confidence thresholding.',
            border='#6a1b9a',
        )
        st.code(_excerpt(narrate_src, 188, 224), language='python')

    with tab3:
        _file_badge('xai_forecast/narrate.py — DeepSeekNarrator.generate()')
        col1, col2 = st.columns([3, 2])
        with col1:
            st.code(_excerpt(narrate_src, 20, 29), language='python')
            st.code(_excerpt(narrate_src, 265, 329), language='python')
        with col2:
            st.markdown('#### MAX_NARRATIVE_TOKENS')
            _note(
                '<code>MAX_NARRATIVE_TOKENS = 800</code> — generous 4× margin above worst-case schema output (~200 tokens). '
                'Named constant so it\'s easy to find and raise if the warning fires.<br><br>'
                'If the API truncates the response, <code>finish_reason == "length"</code> is detected '
                '<i>before</i> attempting json.loads — avoiding a cryptic JSONDecodeError. '
                'Returns None with an actionable log message.',
                border='#6a1b9a',
            )
            st.markdown('#### Graceful no-key fallback')
            _note(
                'If <code>DEEPSEEK_API_KEY</code> is not set, <code>self._client</code> stays None. '
                '<code>generate()</code> returns None immediately. '
                'The dashboard falls back to charts-only, no errors.',
                border='#6a1b9a',
            )
            st.markdown('#### Grounding check')
            _note(
                'After parsing the JSON, validates that <code>primary_driver</code> is in the dossier\'s <code>features</code> list. '
                'If not: sets <code>confidence="low"</code> and <code>grounding_warning=True</code>. '
                'Does NOT validate body text or numbers — low temperature (0.2) + prompt rules handle those.',
                border='#6a1b9a',
            )
            _note(
                '⚠️ <b>Prompt constants rule</b>: the three <code>*_PROMPT</code> constants at the top of narrate.py '
                'must be audited with <code>/prompt-audit</code> before committing any edit.',
                border='#e53935',
            )

    with tab4:
        st.markdown('#### Three prompt constants')
        _file_badge('xai_forecast/narrate.py — WEEK_NARRATIVE_PROMPT / ITEM_NARRATIVE_PROMPT / EXECUTIVE_NARRATIVE_PROMPT')
        st.code(_excerpt(narrate_src, 32, 95), language='python')
        _note(
            'All three prompts share the same pattern:<br>'
            '1. Role framing (retail analyst)<br>'
            '2. Anti-hallucination rule (only use evidence JSON)<br>'
            '3. Exact JSON schema inline<br>'
            '4. Confidence thresholding rule<br>'
            '5. Jargon ban (no "SHAP", "log-margin")<br>'
            '6. Language constraint (English only)<br>'
            '7. Self-review step (verify primary_driver, numbers, jargon before responding)',
            border='#6a1b9a',
        )


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
                'Every run begins with <code>DELETE FROM forecasts/evaluations/xai_results/narratives</code>. '
                'No partial runs, no orphan rows, always a clean consistent result. '
                'INSERT OR REPLACE provides additional idempotency within a run.',
                border='#37474f',
            )
            st.markdown('#### Last two weeks excluded')
            _note(
                '<code>backtest_weeks = weeks[TRAIN_WINDOW:-2]</code> — M5 evaluation file ends 2 days '
                'into the final fiscal week (partial week, artificially low sales). '
                'Forecasting it produces a spurious ~215% MAPE spike. Excluded.',
                border='#e53935',
            )

        st.markdown('#### XAI + narrative phase (after bad-week flagging)')
        st.code(_excerpt(backtest_src, 118, min(200, len(backtest_src.splitlines()))), language='python')

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
            '5. <b>Narrative API probe</b>: one live DeepSeek call. Fails loudly on bad config or truncation.',
            border='#37474f',
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '🗄️  Storage':
    _header('🗄️ Storage', 'db.py · 4 migrations · SQLite WAL', COLORS[section])

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
            st.markdown('#### narratives table')
            _note(
                'PRIMARY KEY (scope, key).<br>'
                'Three scopes: <code>week</code> (key=forecast_week), '
                '<code>item</code> (key=week::item_id), '
                '<code>executive</code> (key="overall").',
                border='#00695c',
            )

    with tab2:
        st.markdown('#### 4 migration files (applied in sorted order)')
        for mig_file in sorted((ROOT / 'migrations').glob('*.sql')):
            with st.expander(mig_file.name):
                st.code(mig_file.read_text(encoding='utf-8'), language='sql')

        st.markdown('#### Table ownership')
        st.dataframe(
            {
                'Table': ['weekly_sales', 'calendar', 'prices', 'item_meta', 'features', 'forecasts', 'evaluations', 'xai_results', 'narratives'],
                'Written by': ['ingest.py', 'ingest.py', 'ingest.py', 'ingest.py', 'build_features.py', 'backtest.py', 'backtest.py', 'backtest.py', 'backtest.py'],
                'Purpose': [
                    'Raw weekly unit sales per SKU',
                    'SNAP, event flags per week',
                    'Weekly avg sell price per SKU',
                    'dept_id, cat_id, mean sales priors',
                    'Precomputed feature matrix (847k rows)',
                    'h=1 predictions per SKU per week',
                    'MAPE, MAE, WMAPE z-score, bad-week flag',
                    'JSON payloads: shap / counterfactual / contrastive',
                    'LLM narratives (scope, key) with model + created_at',
                ],
            },
            width="stretch",
        )

    with tab3:
        st.markdown('#### Why INSERT OR REPLACE everywhere?')
        _note(
            'All four output tables (forecasts, evaluations, xai_results, narratives) use '
            '<code>INSERT OR REPLACE</code> keyed on their primary keys.<br><br>'
            'This makes the backtest idempotent: re-running always produces a clean result. '
            'The clean-slate DELETE at the start is the primary guarantee; INSERT OR REPLACE is '
            'a belt-and-suspenders safety net for partial writes.',
            border='#00695c',
        )
        st.markdown('#### Example: insert_forecasts')
        st.code(_excerpt(db_src, 92, 98), language='python')
        st.markdown('#### Example: insert_narrative')
        st.code(_excerpt(db_src, 143, 150), language='python')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '📊  Dashboard':
    _header('📊 Dashboard', 'app.py — 4-page Streamlit leader view', COLORS[section])

    app_src = _read('app.py')

    st.markdown('**Run:** `uv run streamlit run app.py` → `localhost:7501`')

    st.markdown('#### Pages')
    pages = {
        'Overview': ('lines 150–183', 'Weekly MAPE time series + bad-week markers (×). Four KPI tiles: total weeks, bad weeks, avg MAPE, worst MAPE.'),
        'Bad Week Drilldown': ('lines 188–255', 'LLM week narrative card → worst-30 table → SHAP driver bar chart (mean |SHAP| per feature, % of SKUs labelled) → MAPE distribution histogram.'),
        'Recurring Drivers': ('lines 260–314', 'LLM executive synthesis card → feature frequency bar chart (across all bad weeks) → frequency table with % of payloads.'),
        'XAI Explorer': ('lines 319–504', 'LLM item narrative card → 3 tabs: SHAP waterfall (with residual bar), Counterfactual bar chart (active scenarios only, inactive grayed), Contrastive grouped bar + diff table.'),
    }
    for page_name, (loc, desc) in pages.items():
        with st.expander(f'**{page_name}** — {loc}'):
            st.markdown(desc)

    tab1, tab2, tab3, tab4 = st.tabs(['Data helpers', 'SHAP waterfall', 'Counterfactual', 'Contrastive'])

    with tab1:
        st.markdown('#### Caching strategy')
        st.code(_excerpt(app_src, 38, 103), language='python')
        _note(
            '<code>@st.cache_data</code> on stable reads (week_summary, evaluations, all_shap_payloads). '
            'Narrative helpers are NOT cached — narratives can be regenerated between dashboard opens. '
            'Bare <code>except (sqlite3.OperationalError, json.JSONDecodeError)</code> on narrative helpers '
            '— only catches DB errors and malformed JSON, not generic exceptions.',
            border='#0277bd',
        )

    with tab2:
        st.markdown('#### SHAP waterfall — the key visualisation')
        st.code(_excerpt(app_src, 362, 397), language='python')
        _note(
            'Plotly Waterfall with <code>base=base_log</code>. The "other N features" bar is the residual '
            '(sum of non-top-5 SHAP values) — ensures the waterfall ends exactly at log(prediction). '
            '<code>n_other = len(FEATURE_COLS) - len(feats)</code> (uses FEATURE_COLS constant, not magic 19).',
            border='#0277bd',
        )

    with tab3:
        st.markdown('#### Counterfactual — inactive scenarios grayed')
        st.code(_excerpt(app_src, 399, 455), language='python')
        _note(
            'Only active scenarios are plotted. Inactive scenarios (was_active=False) appear in '
            'the table with ✗ label — visible but clearly not meaningful counterfactuals.',
            border='#0277bd',
        )

    with tab4:
        st.markdown('#### Contrastive — grouped bar: bad vs good week')
        st.code(_excerpt(app_src, 457, 504), language='python')
        _note(
            'Two bar series (red = bad week, blue = good reference) grouped by feature. '
            'The diff table below shows <code>shap_diff = bad_shap − good_shap</code> — '
            'the structural divergence between the two weeks.',
            border='#0277bd',
        )

    st.markdown('---')
    st.markdown('#### Full app.py')
    with st.expander('Show full source'):
        st.code(app_src, language='python')
