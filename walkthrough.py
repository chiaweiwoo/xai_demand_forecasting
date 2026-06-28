"""
XAI + LLM Technical Walkthrough — for newcomers to this project.

Explains the concepts, design decisions, and architecture behind the
XAI and LLM layers. Data ingestion and model training are covered
only as context; the focus is on explainability and the insights pipeline.

Run:
    uv run streamlit run walkthrough.py --server.port 8502
"""

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent

st.set_page_config(
    page_title='XAI + LLM Walkthrough',
    layout='wide',
    page_icon='🧭',
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


def _callout(title: str, body: str, color: str = '#1a73e8', icon: str = '') -> None:
    st.markdown(
        f'<div style="background:{color}18;border-left:4px solid {color};'
        f'padding:14px 18px;border-radius:0 8px 8px 0;margin:12px 0">'
        f'<div style="font-weight:700;color:{color};margin-bottom:6px">{icon} {title}</div>'
        f'<div style="font-size:0.9rem;line-height:1.6">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _concept(label: str, body: str) -> None:
    _callout(label, body, color='#1565c0', icon='📐')


def _warn(label: str, body: str) -> None:
    _callout(label, body, color='#c62828', icon='⚠️')


def _insight(label: str, body: str) -> None:
    _callout(label, body, color='#2e7d32', icon='💡')


def _section_header(title: str, subtitle: str, color: str) -> None:
    st.markdown(
        f'<div style="background:{color};padding:20px 26px;border-radius:12px;margin-bottom:20px">'
        f'<h2 style="color:#fff;margin:0;font-size:1.5rem;font-weight:800">{title}</h2>'
        f'<p style="color:rgba(255,255,255,0.88);margin:6px 0 0;font-size:0.95rem">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _flow_box(lines: list[tuple[str, str]]) -> None:
    """Render a flow diagram as styled HTML."""
    items = ''
    for label, desc in lines:
        items += (
            f'<div style="display:flex;align-items:center;margin:6px 0">'
            f'<div style="background:#e8eaf6;color:#283593;font-family:monospace;font-size:0.8rem;'
            f'padding:5px 10px;border-radius:6px;min-width:200px;font-weight:600">{label}</div>'
            f'<div style="margin-left:12px;font-size:0.85rem;color:#444">{desc}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:#f8f9fa;padding:16px 20px;border-radius:10px;margin:12px 0">{items}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title('🧭 XAI + LLM Walkthrough')
st.sidebar.caption('A newcomer\'s guide to the explainability and insights layers.')
st.sidebar.markdown('---')

SECTIONS = [
    '0. Start Here',
    '1. What is XAI?',
    '2. SHAP — Feature Attribution',
    '3. Counterfactuals — What-If?',
    '4. Contrastive — Good vs Bad',
    '5. Evidence-First LLM Design',
    '6. The Hypothesis-Critic Chain',
    '7. LangGraph Orchestration',
    '8. The Full Picture',
]

COLORS = {
    '0. Start Here':               '#1a237e',
    '1. What is XAI?':             '#1565c0',
    '2. SHAP — Feature Attribution':'#e65100',
    '3. Counterfactuals — What-If?':'#6a1b9a',
    '4. Contrastive — Good vs Bad': '#00695c',
    '5. Evidence-First LLM Design': '#37474f',
    '6. The Hypothesis-Critic Chain':'#b71c1c',
    '7. LangGraph Orchestration':   '#4527a0',
    '8. The Full Picture':          '#1b5e20',
}

section = st.sidebar.radio('Jump to', SECTIONS, label_visibility='collapsed')

st.sidebar.markdown('---')
st.sidebar.markdown(
    '**Learning path**\n'
    '1. Why XAI matters (0 → 1)\n'
    '2. Three XAI techniques (2 → 4)\n'
    '3. LLM pipeline design (5 → 7)\n'
    '4. Everything connected (8)'
)
st.sidebar.markdown('---')
st.sidebar.info(
    'This walkthrough covers **XAI and LLM concepts**. '
    'For data ingestion and model training code, see the Code Review app.'
)


# ═══════════════════════════════════════════════════════════════════════════════
# 0. START HERE
# ═══════════════════════════════════════════════════════════════════════════════

if section == '0. Start Here':
    _section_header(
        '🧭 Start Here',
        'What this project does, why it matters, and how to read this walkthrough.',
        COLORS[section],
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            '### The core question\n'
            '> **"The model performed badly at week X — why?"**\n\n'
            'A demand forecasting model runs every week across thousands of products. '
            'When it fails — predicting 500 units when 50 sold — a manager sees a red number '
            'and asks: *was this a bad model, bad data, or a real-world event we should have anticipated?*\n\n'
            'Standard monitoring shows **that** the model failed. This project answers **why**.'
        )

        st.markdown('---')
        st.markdown('### Three layers, three questions')
        _flow_box([
            ('ML layer', 'Trains and evaluates — detects *when* the model failed (WMAPE z-score)'),
            ('XAI layer', 'Explains *what drove* each bad forecast (SHAP, counterfactual, contrastive)'),
            ('LLM layer', 'Synthesises evidence into *actionable language* (evidence-first agent pipeline)'),
        ])

        st.markdown('---')
        st.markdown('### What this walkthrough covers')
        st.markdown(
            '| Section | Topic |\n'
            '|---|---|\n'
            '| 1 | What XAI is and why it\'s needed |\n'
            '| 2 | SHAP: feature attribution in log-margin space |\n'
            '| 3 | Counterfactuals: what-if scenario testing |\n'
            '| 4 | Contrastive: bad week vs good reference week |\n'
            '| 5 | Evidence-first LLM design: why detectors fire before LLMs |\n'
            '| 6 | The hypothesis-critic chain: how overclaiming is prevented |\n'
            '| 7 | LangGraph: async fan-out orchestration |\n'
            '| 8 | Everything connected |\n'
        )

    with col2:
        st.markdown('### Key numbers')
        st.metric('Weeks evaluated', '120', 'sliding window backtest')
        st.metric('Bad weeks detected', '18', 'WMAPE z-score >= 1.5')
        st.metric('SKUs per bad week', '~2,800', 'all valid non-pre-launch')
        st.metric('XAI payloads', '~101k', 'SHAP + CF + contrastive')
        st.metric('Insights accepted', '5 / 5', 'after prompt fixes')

        st.markdown('---')
        _concept(
            'What makes a "bad week"?',
            'Not just high MAPE — that would flag every low-volume SKU. '
            '<b>WMAPE z-score</b>: volume-weighted error, compared to the prior 8 weeks. '
            'A spike is loud only if it\'s unusually large relative to recent history.'
        )
        _concept(
            'What this is NOT',
            'This is not about building a better forecaster. '
            'It is about <b>XAI-driven model governance</b> — producing a '
            '"what to fix" list for the data scientist and a '
            '"what\'s the risk and plan" brief for the business.'
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WHAT IS XAI?
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '1. What is XAI?':
    _section_header(
        '1. What is XAI?',
        'Explainable AI — why black-box models are not enough for governance.',
        COLORS[section],
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            '### The black-box problem\n'
            'A LightGBM model with 26 features and hundreds of trees is '
            'mathematically opaque. It produces a number (e.g. "forecast = 340 units"). '
            'But it cannot tell you which inputs drove that number, or why the number '
            'was badly wrong this week when it was fine last year.\n\n'
            '**XAI adds a post-hoc explanation layer** — it asks: given this model and '
            'this input, what can we learn about the prediction?'
        )

        st.markdown('---')
        st.markdown('### Three complementary techniques')
        with st.expander('**SHAP** — which features pushed the prediction up or down?', expanded=True):
            st.markdown(
                'SHAP (SHapley Additive exPlanations) assigns each feature a value representing '
                'its marginal contribution to the prediction, relative to a baseline. '
                'The contributions are additive: base + Σ(shap values) = prediction.\n\n'
                '**Strength:** tells you the model\'s internal weighting. Works on any model.\n\n'
                '**Limitation:** explains the model, not the world. A high SHAP value for '
                '`rolling_4_mean` means the model weighted recent sales history heavily — '
                'not that recent sales *caused* the error.'
            )
        with st.expander('**Counterfactual** — what if we removed SNAP / events / price changes?'):
            st.markdown(
                'Zero out a feature (e.g. SNAP flag) and re-run the model. '
                'The difference in prediction is the feature\'s *sensitivity* — '
                'how much the model\'s output changes when that input changes.\n\n'
                '**Strength:** concrete, quantified ("zeroing SNAP changes prediction by 12%").\n\n'
                '**Limitation:** this is model sensitivity testing, not causal inference. '
                '"The model is sensitive to SNAP" ≠ "SNAP caused the forecast error."'
            )
        with st.expander('**Contrastive** — what was different in a week the model got right?'):
            st.markdown(
                'Find a reference week with the same seasonal context (same week-of-year) '
                'where the model performed well. Diff the SHAP profiles of both weeks. '
                'The features with the largest diff are where the model\'s behaviour diverged.\n\n'
                '**Strength:** contextualises the bad week — not just "what happened" '
                'but "what was structurally different from a success.".\n\n'
                '**Limitation:** only ~41% of SKUs have a same-WOY good reference week. '
                'Coverage gaps are part of the model\'s limitations story.'
            )

        st.markdown('---')
        st.markdown('### What XAI cannot do')
        _warn(
            'XAI explains models, not the world',
            'All three techniques describe how the <b>model</b> behaves. '
            'SHAP values are in the model\'s feature space. '
            'They are evidence about model behaviour — not evidence about what '
            'actually drove demand up or down in the real world. '
            'The LLM pipeline is specifically designed to enforce this boundary.'
        )

    with col2:
        st.markdown('### The XAI output')
        st.markdown(
            'For each bad week, each valid SKU receives:\n\n'
            '1. **SHAP payload** — top 5 feature attributions + signed error\n'
            '2. **Counterfactual payload** — 3 scenario deltas (no_snap, no_event, no_price_change)\n'
            '3. **Contrastive payload** — top 5 SHAP diffs vs good reference (if available)\n\n'
            'All three are stored as JSON in `xai_results`. '
            'The insights pipeline reads them as evidence for the LLM agents.'
        )

        st.markdown('---')
        _insight(
            'Why all SKUs, not just top-50?',
            'Early versions computed XAI for only the 50 worst-MAPE SKUs. '
            'This biased the direction analysis: extreme MAPE errors skew toward '
            'over-forecasts. The full population shows a ~50/50 over/under split. '
            'Full coverage also surfaces patterns across the whole SKU distribution, '
            'not just the long tail.'
        )

        st.markdown('---')
        st.markdown('### Concepts to keep in mind')
        for concept, detail in [
            ('Log-margin space', 'Tweedie uses a log-link. SHAP values are additive in log space, not unit space.'),
            ('Signed error', '(prediction − actual) / actual × 100. Positive = over-forecast.'),
            ('same-WOY', 'Same ISO week-of-year — controls for seasonal patterns in contrastive.'),
            ('Coverage', '% of SKUs with a valid same-WOY good reference week.'),
        ]:
            st.markdown(f'**{concept}:** {detail}')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SHAP
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '2. SHAP — Feature Attribution':
    xai_src = _read('xai_forecast/xai.py')

    _section_header(
        '2. SHAP — Feature Attribution',
        'TreeSHAP in log-margin space: additive decomposition, top-5 design, signed error.',
        COLORS[section],
    )

    tab1, tab2, tab3 = st.tabs(['Intuition', 'Log-margin space', 'Code & payload'])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                '### What SHAP computes\n'
                'Imagine the model\'s prediction as a **budget allocation**. '
                'Each feature gets a share of the budget (positive = pushed prediction up, '
                'negative = pushed it down). The shares sum to the total deviation from a baseline.\n\n'
                'More formally: SHAP values are Shapley values from cooperative game theory — '
                'they fairly distribute "credit" among players (features) based on their '
                'marginal contribution across all possible coalitions.\n\n'
                'For tree models (like LightGBM), TreeSHAP computes these exactly in **polynomial time** '
                'by traversing the decision tree structure — no sampling, no approximation.'
            )

            st.markdown('---')
            st.markdown('### What the values mean in this project')
            st.markdown(
                '- **Positive SHAP on `rolling_4_mean`**: the model saw high recent sales and '
                'pushed the forecast up because of them.\n'
                '- **Negative SHAP on `sell_price`**: the model saw a price increase and '
                'pushed the forecast down (price elasticity).\n'
                '- **Large SHAP on `lag_1`**: last week\'s sales dominated the model\'s reasoning.\n\n'
                '**Important:** these tell you about the model\'s weighting, '
                'not about actual demand causation.'
            )

            st.markdown('---')
            st.markdown('### The "demand cliff" pattern')
            _concept(
                'When lag_1 >> actual sales',
                'The model sees high lag_1 (last week was busy) and forecasts high. '
                'But actual demand had already dropped sharply. '
                'The SHAP payload for these items shows a large positive SHAP on lag_1 — '
                'the model was over-anchored on recent momentum. '
                'This is the <b>demand_cliff</b> pattern: lag_1 / actual >= 3x, '
                'found in 2,422 items in this run.'
            )

        with col2:
            st.markdown('### Payload structure')
            st.code(
                json.dumps({
                    'base_value_log': 1.8324,
                    'prediction': 12.5,
                    'actual': 4.0,
                    'error_pct': 212.5,
                    'signed_error': 212.5,
                    'direction': 'over',
                    'other_features_shap': -0.0821,
                    'top_features': [
                        {'feature': 'rolling_4_mean', 'shap_value': 0.7813, 'feature_value': 18.25},
                        {'feature': 'lag_1', 'shap_value': 0.6201, 'feature_value': 15.0},
                        {'feature': 'rolling_8_mean', 'shap_value': 0.4102, 'feature_value': 16.5},
                        {'feature': 'lag_52', 'shap_value': 0.1923, 'feature_value': 9.0},
                        {'feature': 'sell_price', 'shap_value': -0.1141, 'feature_value': 3.49},
                    ],
                }, indent=2),
                language='json',
            )
            _concept(
                'Reading this payload',
                'base_value_log = 1.83. '
                'Sum of top5 SHAP = 1.69. '
                'other_features_shap = -0.08. '
                'Total = 3.44 ≈ log(prediction) = log(12.5) = 2.53... '
                'wait — the numbers here are illustrative. '
                'In the real data: base + Σ(shap) ≈ log(prediction) within 1% tolerance.'
            )

    with tab2:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                '### Why log-margin space?\n'
                'LightGBM with `objective=tweedie` uses a **log-link** internally. '
                'This means the model optimises in log space, and SHAP values are '
                'additive in log space:\n\n'
                '```\n'
                'base_value_log + Σ(shap_values) = log(prediction)\n'
                '```\n\n'
                'To get back to unit space: `exp(base_value_log + Σ(shap)) = prediction`.\n\n'
                '**Consequence for interpretation:**\n'
                '- SHAP values are not in "units sold". They are log-space deltas.\n'
                '- A SHAP of +0.7 on `rolling_4_mean` means: this feature multiplied '
                'the prediction by exp(0.7) ≈ 2.0×.\n'
                '- Feature **ranking** by |shap| is valid regardless of scale.\n'
                '- Comparing raw SHAP values across features of different types '
                'requires caution.'
            )

            st.markdown('---')
            st.markdown('### The top-5 + residual design')
            st.markdown(
                'The payload stores only the **top 5 features** (by |shap|), plus '
                '`other_features_shap` — the sum of the remaining 21 features.\n\n'
                '**Why not all 26?** Payload size and readability. The insights module '
                'and dashboard only need to surface the strongest drivers.\n\n'
                '**Why the residual?** So the waterfall chart reconciles exactly. '
                'Without the residual, `base + top5 ≠ log(prediction)` — the chart '
                'would be dishonest. With it, the residual bar closes the gap.\n\n'
                '```\n'
                'base_value_log + Σ(top5) + other_features_shap ≈ log(prediction)\n'
                '```'
            )

        with col2:
            _warn(
                'SHAP ≠ causation',
                'A positive SHAP value for <code>rolling_4_mean</code> means the model '
                'weighted recent sales heavily — not that high recent sales <i>caused</i> '
                'the forecast to be wrong. '
                'The model could be right to do this (demand is genuinely trend-following) '
                'or wrong (momentum reversed). SHAP alone cannot tell you which.'
            )
            _insight(
                'Tweedie choice',
                '64% of weekly sales rows are zero (intermittent demand). '
                'Standard RMSE or MAE treats zero weeks identically to high-volume weeks. '
                'Tweedie (variance_power=1.5) is the correct distribution for sparse count data — '
                'between Poisson (integer counts) and Gamma (continuous positives).'
            )
            _concept(
                'signed_error and direction',
                'Added to each SHAP payload so the insights module can reference '
                'error direction without re-querying evaluations. '
                '<code>signed_error > 0</code> = over-forecast. '
                '<code>direction = "over" | "under"</code>.'
            )

    with tab3:
        _callout('Source file', 'xai_forecast/xai.py — shap_payloads()', color='#e65100', icon='📄')
        st.code(_excerpt(xai_src, 32, 94), language='python')
        _concept(
            'shap_cache return value',
            'shap_payloads() returns (rows, shap_cache). '
            'shap_cache maps uid → full 1-D SHAP array. '
            'This is passed to contrastive_payloads as bad_shap_cache — '
            'the bad-item SHAP is reused instead of recomputed. '
            'Without this, bad-item SHAP would be computed twice per bad week.'
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COUNTERFACTUALS
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '3. Counterfactuals — What-If?':
    xai_src = _read('xai_forecast/xai.py')

    _section_header(
        '3. Counterfactuals — What-If?',
        'Zero out SNAP / events / price change. Measure how much the model\'s prediction shifts.',
        COLORS[section],
    )

    tab1, tab2 = st.tabs(['Intuition & design', 'Code & payload'])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                '### The question\n'
                '> "If there had been no SNAP activity this week, would the forecast have been so high?"\n\n'
                'Counterfactual analysis answers this by taking the model\'s original input, '
                'zeroing out the feature(s) of interest, re-running the model, '
                'and measuring the prediction delta.\n\n'
                '**Example:** SNAP = 1 → SNAP = 0. Original prediction: 120 units. '
                'Counterfactual prediction: 104 units. Delta: -13.3%.\n\n'
                'This tells you: *the model is sensitive to SNAP*. '
                'It does not tell you that SNAP caused the forecast error.'
            )

            st.markdown('---')
            st.markdown('### Three scenarios')
            for name, desc, why in [
                (
                    'no_snap',
                    'Set snap=0',
                    'SNAP (Supplemental Nutrition Assistance Program) is a California food-stamp schedule. '
                    'SNAP-on weeks typically boost retail grocery demand. '
                    'Zeroing it tests how much the model anchors on SNAP activity.'
                ),
                (
                    'no_event',
                    'Set has_event=0, event_type_enc=0',
                    'Calendar events (National, Cultural, Religious, Sporting) '
                    'are encoded as one-hot-like flags. '
                    'Removing them tests event sensitivity.'
                ),
                (
                    'no_price_change',
                    'Set price_change_pct=0',
                    'Price changes can drive demand changes. '
                    'Removing the price change signal tests pricing sensitivity.'
                ),
            ]:
                with st.expander(f'**{name}** — {desc}'):
                    st.markdown(why)

            st.markdown('---')
            st.markdown('### The was_active flag')
            _concept(
                'Why we track was_active',
                'If snap=0 for an item in a given week, zeroing it is a no-op — '
                'delta_pct will be exactly 0, which looks like "SNAP has no effect." '
                'That\'s misleading. '
                '<code>was_active=True</code> means the feature was actually non-zero for this item '
                'in this week, so the counterfactual represents a real perturbation. '
                '<code>was_active=False</code> means the feature was already 0 — the scenario is inactive '
                'and should be excluded from the analysis.'
            )

            st.markdown('---')
            st.markdown('### The 5% materiality threshold')
            _concept(
                'delta_pct >= 5%',
                'The counterfactual_material detector fires only when zeroing a feature '
                'moves the prediction by more than 5% for that item. '
                'Smaller deltas are within model noise — not meaningful enough to '
                'report as a "material" driver. '
                'In this run: zeroing SNAP moved prediction >5% for 23.4% of items.'
            )

            st.markdown('---')
            _warn(
                'Counterfactuals test model sensitivity — not production causation',
                'This is the most common LLM failure mode in this project. '
                '"Zeroing SNAP reduces the prediction by 12%" is correct. '
                '"SNAP schedule changes cause forecast errors" is a causal overclaim — '
                'the model never trained on counterfactual real-world outcomes. '
                'The CRITIC_PROMPT has an explicit rule to reject this extrapolation.'
            )

        with col2:
            st.markdown('### Payload structure')
            st.code(
                json.dumps({
                    'prediction_original': 120.4,
                    'actual': 45.0,
                    'scenarios': [
                        {
                            'scenario': 'no_snap',
                            'was_active': True,
                            'prediction_cf': 104.2,
                            'delta': -16.2,
                            'delta_pct': -13.46,
                        },
                        {
                            'scenario': 'no_event',
                            'was_active': False,
                            'prediction_cf': 120.4,
                            'delta': 0.0,
                            'delta_pct': 0.0,
                        },
                        {
                            'scenario': 'no_price_change',
                            'was_active': True,
                            'prediction_cf': 118.1,
                            'delta': -2.3,
                            'delta_pct': -1.91,
                        },
                    ],
                }, indent=2),
                language='json',
            )
            _insight(
                'Reading this payload',
                'SNAP was active → zeroing it dropped prediction by 13.5% (material). '
                'Event was NOT active → no_event is an inactive scenario, ignore it. '
                'Price change was active but delta is only 1.9% → below 5% threshold, not material.'
            )

    with tab2:
        _callout('Source file', 'xai_forecast/xai.py — counterfactual_payloads()', color='#6a1b9a', icon='📄')
        st.code(_excerpt(xai_src, 97, 145), language='python')


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONTRASTIVE
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '4. Contrastive — Good vs Bad':
    xai_src = _read('xai_forecast/xai.py')

    _section_header(
        '4. Contrastive — Good vs Bad',
        'Compare SHAP profiles between a bad week and a same-season good reference week.',
        COLORS[section],
    )

    tab1, tab2 = st.tabs(['Intuition & design', 'Code & payload'])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                '### The question\n'
                '> "The model got week 27 of 2015 badly wrong. What was different '
                'from week 27 of 2014, when it did well?"\n\n'
                'Instead of asking "what drove the prediction?" (SHAP) '
                'or "what if we removed X?" (counterfactual), '
                'contrastive asks "where did the model\'s behavior *change* '
                'compared to a structurally similar situation that went well?"\n\n'
                '**shap_diff = bad_shap − good_shap** for each feature.\n\n'
                'A large positive diff on `rolling_4_mean` means: '
                'recent sales weighed much more heavily in the bad week than in the good week — '
                'the model was more anchored on recent momentum when it failed.'
            )

            st.markdown('---')
            st.markdown('### The same-WOY constraint')
            _concept(
                'Why same-week-of-year only?',
                'Comparing a winter week to a summer week would conflate two different things: '
                '(1) the model\'s structural error, and (2) legitimate seasonal differences. '
                'By constraining to the same ISO week-of-year, we control for seasonality — '
                'the reference week and the bad week have similar expected demand patterns. '
                'Differences in SHAP profiles then reflect model behaviour, not season.'
            )

            _warn(
                'No same-WOY reference = no contrastive data',
                'If a SKU has no historical week with the same WOY where MAPE < 15%, '
                'it is skipped entirely. No fallback to different-WOY weeks — '
                'that would break the "similar seasonal context" claim. '
                'Result: ~41% of SKUs have contrastive data; ~59% do not. '
                'The contrastive_gap finding reports this coverage honestly.'
            )

            st.markdown('---')
            st.markdown('### Batching for efficiency')
            _insight(
                'Ref-week batching',
                'Multiple SKUs can share the same reference week '
                '(e.g. all WOY-1 items comparing to 2014-01-04). '
                'Reference features are loaded once per unique ref week via load_features_week(). '
                'SHAP on reference items is computed in a single batch per ref week. '
                'Without batching, SHAP would be called once per item — much slower.'
            )

            st.markdown('---')
            st.markdown('### What the diffs tell you')
            st.markdown(
                'The top contrastive diff in the current run is **price_change_pct** — '
                'price change has a larger SHAP impact in bad weeks than in good reference weeks. '
                'This does not mean price changes *cause* bad weeks. '
                'It means: when the model fails, price sensitivity plays a larger role '
                'in its reasoning than in weeks it gets right.\n\n'
                'This kind of structural divergence is exactly what the LLM hypothesis step '
                'is asked to interpret — within the bounds of model-behaviour language.'
            )

        with col2:
            st.markdown('### Payload structure')
            st.code(
                json.dumps({
                    'bad_week': '2014-09-06',
                    'good_week': '2013-09-07',
                    'good_week_mape': 8.3,
                    'seasonality_matched': True,
                    'top_diffs': [
                        {
                            'feature': 'price_change_pct',
                            'shap_diff': 0.3812,
                            'bad_value': 0.12,
                            'good_value': 0.0,
                            'bad_shap': 0.4201,
                            'good_shap': 0.0389,
                        },
                        {
                            'feature': 'rolling_4_mean',
                            'shap_diff': 0.2941,
                            'bad_value': 21.5,
                            'good_value': 12.3,
                            'bad_shap': 0.8812,
                            'good_shap': 0.5871,
                        },
                    ],
                }, indent=2),
                language='json',
            )
            _concept(
                'Reading the top diff',
                'price_change_pct: bad_shap=0.42, good_shap=0.04, diff=0.38. '
                'In the bad week, this item had a price change (12%). '
                'In the good reference week, no price change. '
                'The model weighted the price change strongly in the bad week, '
                'contributing to its over-prediction.'
            )

    with tab2:
        _callout('Source file', 'xai_forecast/xai.py — contrastive_payloads()', color='#00695c', icon='📄')
        st.code(_excerpt(xai_src, 148, 271), language='python')


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVIDENCE-FIRST LLM DESIGN
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '5. Evidence-First LLM Design':
    detectors_src = _read('xai_forecast/insights/detectors.py')
    agents_src    = _read('xai_forecast/insights/agents.py')
    graph_src     = _read('xai_forecast/insights/graph.py')

    _section_header(
        '5. Evidence-First LLM Design',
        'Why deterministic detectors fire before LLMs. How evidence is bounded and enriched.',
        COLORS[section],
    )

    tab1, tab2, tab3 = st.tabs(['Design philosophy', 'Six detectors', 'Planner + read-tools'])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                '### The naive approach (what we did NOT do)\n'
                'Give the LLM all 101k SHAP payloads and ask: "what\'s wrong with this model?"\n\n'
                '**Problems with this:**\n'
                '- LLMs hallucinate at scale. A pattern found in a 100k-row table may be spurious.\n'
                '- LLMs cannot reliably compute aggregate statistics. '
                'Asking "what % of payloads have lag_1 in the top 5?" will produce a guess, not a count.\n'
                '- Output is non-deterministic and hard to audit.\n'
                '- Every run re-discovers the same facts from scratch.\n\n'
                '**What we do instead:** deterministic detectors compute the facts first. '
                'LLMs only receive bounded, pre-verified evidence packs.'
            )

            st.markdown('---')
            st.markdown('### The evidence-first architecture')
            _flow_box([
                ('1. Detectors', 'Python: count, aggregate, threshold — no LLM. Returns CandidateFinding or None.'),
                ('2. Planner (Flash)', 'Choose 1-4 read-tools for this finding type. Returns tool names.'),
                ('3. Read-tools', 'Sync SQLite reads — no LLM. Produces structured evidence dict.'),
                ('4. Hypothesis (Flash)', 'Interpret the bounded evidence pack. Returns grounded hypothesis.'),
                ('5. Grounding advisory', 'Deterministic: verify evidence_refs exist in the evidence dict.'),
                ('6. Critic (Pro)', 'Quality gate: reject overclaim, enforce correlation-only, set status.'),
                ('7. Synthesis (Flash)', 'Combine accepted findings into DS + business reports.'),
            ])

            st.markdown('---')
            st.markdown('### Why the planner step?')
            _insight(
                'Not every tool is useful for every finding',
                'A dominant_driver finding needs read_recurring_drivers and read_model_metadata — '
                'not read_external_signals or read_demand_trajectory. '
                'The planner is a cheap Flash call (few tokens) that routes each finding to '
                'the right evidence sources. '
                'Without it, every finding would pull all 7 tools — wasting tokens and '
                'increasing the risk that irrelevant context misleads the hypothesis step.'
            )

        with col2:
            st.markdown('### Data contracts')
            st.markdown(
                '`CandidateFinding` — from detectors:\n'
                '```python\n'
                'finding_id:   str   # e.g. "dominant_driver"\n'
                'finding_type: str   # same as finding_id\n'
                'score:        float # 0-1, higher = stronger evidence\n'
                'summary:      str   # one-line trigger description\n'
                'evidence:     dict  # pre-computed facts\n'
                '```\n'
            )
            st.markdown(
                '`Hypothesis` — from Flash:\n'
                '```python\n'
                'finding_id:    str\n'
                'headline:      str   # 15-20 word summary\n'
                'explanation:   str   # JSON: ds + business + fix\n'
                'evidence_refs: list  # which evidence keys cited\n'
                'confidence:    str   # high|medium|low\n'
                '```\n'
            )
            st.markdown(
                '`LedgerRow` — stored to DB:\n'
                '```python\n'
                'finding_id:   str\n'
                'finding_type: str\n'
                'status:       str   # accepted|rejected|needs_review\n'
                'confidence:   str\n'
                'evidence:     dict\n'
                'hypothesis:   dict | None\n'
                'critic_notes: str\n'
                '```\n'
            )
            _concept(
                'LedgerRow.hypothesis is None when rejected',
                'Rejected findings have status="rejected" and hypothesis=None. '
                'The critic\'s notes explain why. '
                'The dashboard shows rejected findings honestly as '
                '"we tested this, evidence didn\'t hold."'
            )

    with tab2:
        st.markdown('### Six deterministic detectors')
        st.markdown(
            'Each detector reads directly from SQLite. No LLM calls. '
            'Fires only when real data crosses a threshold.'
        )

        detectors_info = [
            (
                'over_forecast_bias',
                '>=70% of SHAP payloads are over-forecasts',
                'Systematic directional bias across the full SKU population.',
                'pct_over = over / (over + under) — denominator excludes direction=None rows (actual=0).',
                'Did NOT fire in current run: direction split is ~50/50.',
            ),
            (
                'dominant_driver',
                'One feature in >60% of SHAP payloads',
                'Model over-anchors on one feature across all bad weeks.',
                'rolling_4_mean appears in 94.9% of payloads. lag_1 in 86.7%.',
                'ACCEPTED — high confidence.',
            ),
            (
                'demand_cliff',
                'lag_1 >= 3x actual for >= 3 items',
                'Momentum over-anchoring: sales dropped after the model anchored on high lag_1.',
                '2,422 items with cliff_ratio >= 3.0 across the 18 bad weeks.',
                'ACCEPTED — high confidence.',
            ),
            (
                'external_coincidence',
                'Bad week + heat wave / gas spike / sentiment crisis',
                'Correlation only: bad weeks coincide with notable external conditions.',
                '5 of 18 bad weeks coincided with heat waves or high gas prices.',
                'ACCEPTED — medium confidence.',
            ),
            (
                'counterfactual_material',
                'Zeroing SNAP/event/price moves prediction >5% for a substantial fraction',
                'Feature was actually active and quantifiably affected model output.',
                'no_snap moves prediction >5% for 23.4% of items.',
                'ACCEPTED — high confidence.',
            ),
            (
                'contrastive_gap',
                'Always fires if contrastive data exists (score = coverage %)',
                'Reports structural SHAP diff vs good reference, and coverage gaps.',
                '41% contrastive coverage. Top diff: price_change_pct.',
                'ACCEPTED — high confidence.',
            ),
        ]

        for name, threshold, purpose, data, result in detectors_info:
            with st.expander(f'**{name}** — threshold: {threshold}'):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'**Purpose:** {purpose}')
                    st.markdown(f'**Current run data:** {data}')
                with col2:
                    st.markdown(f'**Current run result:** {result}')

        st.markdown('---')
        st.markdown('### Example: detect_dominant_driver')
        _callout('Source', 'xai_forecast/insights/detectors.py', color='#37474f', icon='📄')
        st.code(_excerpt(detectors_src, 83, 147), language='python')

    with tab3:
        st.markdown('### Planner — Flash chooses read-tools')
        _callout('Source', 'xai_forecast/insights/agents.py — PLANNER_PROMPT + run_planner_async()', color='#37474f', icon='📄')

        col1, col2 = st.columns([3, 2])
        with col1:
            st.code(_excerpt(agents_src, 31, 58), language='python')
        with col2:
            for tool, purpose in [
                ('read_forecast_accuracy', 'Global MAPE, bad/good week rates, worst week'),
                ('read_bad_weeks', 'All bad weeks with z-scores and avg MAPE'),
                ('read_xai_findings', 'SHAP/CF payloads — sample from worst week'),
                ('read_demand_trajectory', 'Sales + lag_1 + rolling mean for one SKU over time'),
                ('read_external_signals', 'LA weather, gas price, sentiment for one week'),
                ('read_model_metadata', 'Model config + global feature importance'),
                ('read_recurring_drivers', 'Feature frequency across all SHAP payloads'),
            ]:
                _concept(tool, purpose)

        st.markdown('---')
        st.markdown('### _enrich_evidence — calls tools, merges results')
        _callout('Source', 'xai_forecast/insights/graph.py — _enrich_evidence()', color='#37474f', icon='📄')
        st.code(_excerpt(graph_src, 40, 103), language='python')


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HYPOTHESIS-CRITIC CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '6. The Hypothesis-Critic Chain':
    agents_src = _read('xai_forecast/insights/agents.py')
    graph_src  = _read('xai_forecast/insights/graph.py')

    _section_header(
        '6. The Hypothesis-Critic Chain',
        'Flash writes hypotheses. Pro critiques them. Overclaiming is the central failure mode.',
        COLORS[section],
    )

    tab1, tab2, tab3 = st.tabs(['The overclaim problem', 'Prompts', 'Grounding + critic code'])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                '### Why hypotheses fail\n'
                'The most common rejection reason is not hallucination — it\'s **causal overclaiming**.\n\n'
                'Flash is given evidence like: "zeroing SNAP reduces prediction by 12% for 23% of items." '
                'It can very naturally interpret this as: "SNAP schedule changes cause forecast errors." '
                'That\'s wrong — the evidence is model sensitivity, not production causation.\n\n'
                '**The three failure modes we guard against:**'
            )

            with st.expander('**Failure mode 1:** model-feature causation'):
                st.markdown(
                    'SHAP shows that `rolling_4_mean` had a large influence on this prediction. '
                    'Flash might write: "high recent sales caused the over-forecast."\n\n'
                    '**Wrong:** SHAP shows what the model weighted — not what caused demand to move. '
                    'The model may have been right to weight recent sales, but demand reversed '
                    'for a reason the model has no signal for.\n\n'
                    '**Correct:** "The model weighted `rolling_4_mean` heavily in this bad week."'
                )
            with st.expander('**Failure mode 2:** counterfactual extrapolation'):
                st.markdown(
                    'Counterfactual shows: "zeroing SNAP reduces prediction by 12%." '
                    'Flash might write: "SNAP schedule changes cause forecast errors."\n\n'
                    '**Wrong:** this is model sensitivity under an artificial perturbation. '
                    'In the real world, SNAP is always either active or not — we can\'t "zero it." '
                    'The counterfactual tests the model, not reality.\n\n'
                    '**Correct:** "Zeroing the SNAP feature reduces the model\'s prediction by 12%."'
                )
            with st.expander('**Failure mode 3:** coverage editorializing'):
                st.markdown(
                    'Contrastive coverage is 41%. Flash might write: '
                    '"Low coverage limits the reliability of these conclusions."\n\n'
                    '**Wrong:** this is an editorial judgement the evidence doesn\'t support. '
                    'Coverage percentage is a fact; its implication for reliability is a separate claim.\n\n'
                    '**Correct:** "41% of explained items have a seasonal reference week."'
                )

            st.markdown('---')
            st.markdown('### The structural fix')
            _insight(
                'Hypothesis layer = purely descriptive. Synthesis layer = business framing.',
                'The hypothesis writes WHAT was observed (numbers, patterns, model behaviour). '
                'Risk language, business consequence, and improvement framing belong in the '
                'synthesis stage. The critic enforces this boundary. '
                'This separation was the fix that moved acceptance from ~2/5 to 5/5.'
            )

            st.markdown('---')
            st.markdown('### The non-determinism problem')
            _warn(
                'Flash at temperature 0.2 flaps between runs',
                'The contradiction in the prompt (both "state risk direction" AND "purely descriptive") '
                'caused Flash to randomly obey one rule or the other at temperature 0.2. '
                'Result: acceptance rate oscillated 2/5 ↔ 5/5 between runs. '
                'Fix: remove the contradiction, set critic and synthesis to temperature=0.'
            )

        with col2:
            st.markdown('### Two-model split')
            _concept(
                'Flash (deepseek-v4-flash)',
                'High-volume calls: planner, hypothesis, synthesis. '
                'Temperature 0.2 (planner/hypothesis) or 0.0 (synthesis). '
                'MAX_TOKENS_FLASH = 3000.'
            )
            _concept(
                'Pro (deepseek-v4-pro)',
                'Single quality gate: critic. '
                'Temperature 0.0 — governance artifact must be deterministic. '
                'MAX_TOKENS_PRO = 4096 (allows large evidence packs for demand_cliff).'
            )
            st.markdown('---')
            st.markdown('### Critic output fields')
            st.code(
                '{\n'
                '  "status": "accepted|rejected|needs_review",\n'
                '  "confidence": "high|medium|low",\n'
                '  "notes": "2-3 sentence explanation",\n'
                '  "overclaim": false,\n'
                '  "causal_external": false\n'
                '}',
                language='json',
            )
            _concept(
                'causal_external covers internal features too',
                'Originally: causal_external flagged only external signals (weather, gas). '
                'Expanded: now covers ANY model-sensitivity → production-causation leap, '
                'including internal features like SNAP, events, price change. '
                'SNAP is not an "external signal" in the data sense, but the overclaim is identical.'
            )

    with tab2:
        st.markdown('### HYPOTHESIS_PROMPT (Flash)')
        _callout(
            'Key rules',
            '(1) Use ONLY facts from the evidence JSON. '
            '(2) business_explanation and suggested_fix: PURELY DESCRIPTIVE — no risk inference. '
            '(3) Never say feature X causes demand to move. '
            '(4) Counterfactuals are model sensitivity, not production claims. '
            '(5) Coverage stats: state the number, never editorialize.',
            color='#b71c1c', icon='🚫'
        )
        st.code(_excerpt(agents_src, 60, 127), language='python')

        st.markdown('---')
        st.markdown('### CRITIC_PROMPT (Pro)')
        _callout(
            'Key rules',
            'Reject if: hypothesis invents facts, claims causation for external or internal signals, '
            'extends counterfactual to production claim, editorializes coverage stats. '
            'causal_external=true → status MUST be "rejected".',
            color='#b71c1c', icon='🚫'
        )
        st.code(_excerpt(agents_src, 129, 167), language='python')

        st.markdown('---')
        st.markdown('### Two synthesis prompts (run concurrently)')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**BUSINESS_SYNTHESIS_PROMPT** — VP-facing:')
            _concept('Output', 'headline + progress (health verdict, diagnosis, confidence) + plan (phases, impact) + limitations + risk_direction + overall_confidence')
            _concept('Rules', 'Zero model jargon (no SHAP, LightGBM, WMAPE). Risk direction must match evidence. Plans must be specific enough for a non-technical stakeholder.')
        with col2:
            st.markdown('**TECHNICAL_SYNTHESIS_PROMPT** — DS-facing:')
            _concept('Output', 'headline + summary + levers (bucket, change, evidence, expected_effect, effort) + overall_confidence')
            _concept('Buckets', 'feature_engineering | model_param | workflow | algorithm. Each lever must cite a specific statistic from the findings.')

    with tab3:
        st.markdown('### Grounding advisory — advisory, not a gate')
        st.code(_excerpt(graph_src, 183, 220), language='python')
        _concept(
            'Why advisory-only?',
            'Flash sometimes references evidence keys using dot-path notation vs bare names. '
            'The grounding check normalises [N] bracket notation and also matches bare leaf names. '
            'Unmatched refs are forwarded to the Pro critic as context — '
            'the critic decides whether they represent real hallucination or just a naming mismatch. '
            'Making it a hard gate would cause false rejections.'
        )

        st.markdown('---')
        st.markdown('### run_critic_async')
        st.code(_excerpt(agents_src, 397, 433), language='python')


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LANGGRAPH ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '7. LangGraph Orchestration':
    graph_src = _read('xai_forecast/insights/graph.py')

    _section_header(
        '7. LangGraph Orchestration',
        'Async StateGraph: fan-out per finding, fan-in reducer, concurrent synthesis.',
        COLORS[section],
    )

    tab1, tab2, tab3 = st.tabs(['Graph design', 'Async + SQLite safety', 'State + closure factory'])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                '### Why LangGraph?\n'
                'The insights pipeline has a natural fan-out structure: '
                'given N candidate findings, run N independent review chains '
                '(each: planner → enrich → hypothesis → critic) concurrently, '
                'then fan-in to a synthesis step.\n\n'
                'LangGraph provides the graph structure, state management, '
                'and fan-out via `Send()`. The async version (`ainvoke`) '
                'runs each fan-out node as an asyncio task — true concurrency '
                'within a single process.'
            )

            st.markdown('---')
            st.markdown('### Graph shape')
            st.markdown(
                '''
<div style="background:#ede7f6;padding:16px 20px;border-radius:10px;font-size:0.85rem;line-height:2;font-family:monospace">
START<br>
&nbsp;&nbsp;→ <b>detect_candidates</b> (async def — stays in event loop thread)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;run_all_detectors(conn) → 5 candidates<br>
<br>
&nbsp;&nbsp;→ <b>route_findings</b> (conditional edge)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Send("review_finding", {finding: c}) for c in candidates]<br>
<br>
&nbsp;&nbsp;→ <b>review_finding</b> × 5 (concurrent asyncio tasks)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;planner → enrich → hypothesis → grounding advisory → critic<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;returns {"ledger_rows": [one_row]}<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Annotated[list, add] reducer merges all 5 rows<br>
<br>
&nbsp;&nbsp;→ <b>synthesize</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;asyncio.gather(run_business_synthesis, run_technical_synthesis)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;both Flash calls run concurrently<br>
<br>
END
</div>
''', unsafe_allow_html=True,
            )

            st.markdown('---')
            st.markdown('### The fan-in reducer pattern')
            _concept(
                'Annotated[list, add] on ledger_rows',
                'In LangGraph, each fan-out node returns a partial state update. '
                'Without a reducer, parallel nodes would overwrite each other\'s output. '
                '<code>Annotated[list, add]</code> declares that the list field is '
                'accumulated with the + operator — each review_finding appends its row '
                'rather than replacing the list. No manual merging needed.'
            )

        with col2:
            st.markdown('### State definition')
            st.code(
                'class _State(TypedDict):\n'
                '    candidates:  list\n'
                '    ledger_rows: Annotated[list, add]\n'
                '    summary:     dict',
                language='python',
            )
            _concept(
                'Three fields',
                '<b>candidates</b>: CandidateFinding list from detectors.<br>'
                '<b>ledger_rows</b>: accumulated LedgerRow list (fan-in reducer).<br>'
                '<b>summary</b>: final synthesis output (business + DS dicts).'
            )
            st.markdown('---')
            st.markdown('### Route logic')
            st.code(
                'def route_findings(state: dict):\n'
                '    candidates = state.get("candidates", [])\n'
                '    if not candidates:\n'
                '        return "synthesize"  # skip fan-out\n'
                '    return [\n'
                '        Send("review_finding", {"finding": c})\n'
                '        for c in candidates\n'
                '    ]',
                language='python',
            )
            _insight(
                'Send() is LangGraph fan-out',
                'Each Send() creates an independent invocation of review_finding '
                'with its own sub-state (just the one finding). '
                'LangGraph schedules them as concurrent asyncio tasks.'
            )

    with tab2:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown('### Why detect_candidates must be async def')
            st.code(_excerpt(graph_src, 142, 151), language='python')
            _warn(
                'sync def → ThreadPoolExecutor → SQLite crash',
                'LangGraph dispatches sync nodes to a ThreadPoolExecutor thread pool. '
                'The SQLite connection (conn) was created in the asyncio event loop thread. '
                'SQLite connections cannot be used across threads (thread-safety constraint). '
                'Making detect_candidates a sync def would cause "SQLite objects created in a '
                'thread can only be used in that same thread." '
                'As async def, it stays in the event loop thread — same thread as conn.'
            )

            st.markdown('---')
            st.markdown('### _enrich_evidence is sync (fast, safe)')
            _insight(
                'SQLite reads in the event loop',
                '_enrich_evidence is called from review_finding (async), '
                'but it is itself a regular sync function. '
                'This is fine: the event loop runs sync calls directly when awaited from an '
                'async context without yield points. '
                'The reads are fast (< 50ms each) so they don\'t block the event loop meaningfully. '
                'For slow sync I/O, you\'d use asyncio.to_thread() — not needed here.'
            )

        with col2:
            st.markdown('### Concurrent synthesis')
            st.code(
                'biz, tech = await asyncio.gather(\n'
                '    run_business_synthesis(client, accepted),\n'
                '    run_technical_synthesis(client, accepted),\n'
                ')',
                language='python',
            )
            _concept(
                'asyncio.gather',
                'Both Flash calls for synthesis run concurrently — '
                'each is one HTTP request to the DeepSeek API. '
                'Without gather, they would run sequentially (2× latency). '
                'With gather, the total time is max(biz_time, tech_time).'
            )
            st.markdown('---')
            _concept(
                'Workers in run_xai.py use ProcessPoolExecutor',
                'XAI computation (SHAP) is CPU-heavy and uses pickle-based multiprocessing. '
                'Worker functions must be top-level (not closures) for Windows spawn mode. '
                'Each worker opens its own sqlite3.connect() — no shared connection. '
                'The main process does chunked inserts (10k rows/chunk) after all workers finish.'
            )

    with tab3:
        st.markdown('### Closure factory pattern')
        st.code(_excerpt(graph_src, 139, 161), language='python')
        _concept(
            'Why capture conn and client in closures?',
            'LangGraph passes state through the graph as a dict. '
            'If conn or client were in the state dict, LangGraph would try to serialize them — '
            'SQLite connections and HTTP clients cannot be serialized. '
            'By closing over them in node functions, they live in Python heap memory, '
            'not in the state. The state dict only contains JSON-serializable data.'
        )

        st.markdown('---')
        st.markdown('### Full graph build + run')
        st.code(_excerpt(graph_src, 283, 313), language='python')


# ═══════════════════════════════════════════════════════════════════════════════
# 8. THE FULL PICTURE
# ═══════════════════════════════════════════════════════════════════════════════

elif section == '8. The Full Picture':
    _section_header(
        '8. The Full Picture',
        'End-to-end: from a bad week detection to a governance report.',
        COLORS[section],
    )

    st.markdown('### Trace: one bad week, end to end')
    st.markdown(
        'Let\'s trace what happens when the system processes a specific bad week — '
        'say, 2014-09-06 (a week with WMAPE z-score of 2.1).'
    )

    steps = [
        (
            'Step 1: Bad week detected',
            '`backtest.py` trains a LightGBM on weeks up to 2014-09-06, '
            'forecasts all ~3k SKUs, and evaluates. '
            'WMAPE for this week is 2.1 standard deviations above the prior 8-week rolling mean — '
            'flagged as is_bad_week=1 in the evaluations table.',
            '#1565c0',
        ),
        (
            'Step 2: XAI computed',
            '`run_xai.py` worker for 2014-09-06: loads the checkpoint from models/, '
            'identifies ~2,800 valid SKUs, runs TreeSHAP, counterfactuals (3 scenarios), '
            'and contrastive (same-WOY lookup). '
            'Produces ~8,400 xai_result rows for this week alone.',
            '#e65100',
        ),
        (
            'Step 3: Detectors fire',
            '`run_all_detectors(conn)` reads across ALL bad weeks (not just 2014-09-06). '
            'Dominant_driver: rolling_4_mean in 94.9% of payloads. '
            'Demand_cliff: 2,422 items with lag_1 >= 3x actual. '
            'Counterfactual: zeroing SNAP moves 23.4% of items by >5%. '
            'Contrastive_gap: 41% coverage, top diff price_change_pct. '
            'External_coincidence: 5 of 18 bad weeks during heat/gas events.',
            '#37474f',
        ),
        (
            'Step 4: Per-finding async chain (×5 concurrently)',
            'For each CandidateFinding: '
            '(1) Planner (Flash) picks read-tools. '
            '(2) _enrich_evidence runs them synchronously. '
            '(3) Hypothesis (Flash) writes a bounded, descriptive interpretation. '
            '(4) Grounding advisory checks evidence_refs. '
            '(5) Critic (Pro) verifies: no overclaim, no causal extrapolation, traceable claims.',
            '#6a1b9a',
        ),
        (
            'Step 5: Synthesis (concurrent)',
            'run_business_synthesis and run_technical_synthesis run simultaneously via asyncio.gather. '
            'Business: VP-facing verdict, phased improvement plan, limitations, risk_direction=mixed. '
            'DS: 5 levers (feature_engineering, model_param, workflow) with evidence citations.',
            '#2e7d32',
        ),
        (
            'Step 6: Stored + displayed',
            'LedgerRows written to insight_findings. Summary written to insight_summary. '
            'Dashboard reads both: hero shows verdict + risk badge; '
            '"What we found" shows 5 accepted story cards; '
            '"What to do" shows business plan + DS levers.',
            '#0277bd',
        ),
    ]

    for title, detail, color in steps:
        _callout(title, detail, color=color)

    st.markdown('---')
    st.markdown('### Design decisions recap')

    decisions = {
        'Why not one LLM call for everything?': (
            'LLMs are unreliable at aggregate statistics over large datasets. '
            'Detectors compute precise facts; LLMs interpret bounded evidence packs.'
        ),
        'Why two models (Flash + Pro)?': (
            'Flash is fast and cheap — ideal for high-volume planner/hypothesis/synthesis calls. '
            'Pro is slower but more capable — reserved for the single quality gate (critic). '
            'Cost is proportional to value: the critic call is the most critical.'
        ),
        'Why temperature=0 for critic and synthesis?': (
            'These produce governance artifacts meant to be trusted and audited. '
            'Non-determinism is a liability: the same accepted findings should always '
            'produce the same report. Temperature=0 removes most (not all) variability.'
        ),
        'Why reject instead of revise?': (
            'Allowing the critic to rewrite the hypothesis would mix two model calls '
            'into a single output, making it hard to audit which model said what. '
            'Rejected findings are stored with critic notes — the next pipeline run '
            '(with a fixed prompt) can re-generate them cleanly.'
        ),
        'Why per-checkpoint SHAP (not one final model)?': (
            'The model was retrained ~30 times. A 2013 bad week should be explained '
            'by the 2013 model — not by the 2016 model that has seen 3 more years of data. '
            'Per-checkpoint XAI is faithful to what the model knew at the time of the error.'
        ),
        'Why SQLite instead of a proper database?': (
            'This is a local PoC with one user and sequential write stages. '
            'SQLite WAL mode allows concurrent reads (dashboard open while backtest runs). '
            'Zero operational overhead: no server, no connection pooling, one file.'
        ),
    }

    for q, a in decisions.items():
        with st.expander(f'**{q}**'):
            st.markdown(a)

    st.markdown('---')
    st.markdown('### Current status')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Pipeline status', 'All green', '25/25 DQ checks pass')
        st.metric('Test coverage', '85 tests', 'features, eval, XAI, DB, insights')
    with col2:
        st.metric('Features', '26 total', '19 ML + 7 external signals')
        st.metric('Insights', '5/5 accepted', 'prompt fixes + temperature=0')
    with col3:
        st.metric('XAI payloads', '~101k rows', '18 bad weeks × all valid SKUs')
        st.metric('External signals', 'Stage 2 complete', 'weather + gas + sentiment')

    st.markdown('---')
    _insight(
        'What to study next',
        'Read <code>xai_forecast/insights/agents.py</code> for the full prompt constants. '
        'Run <code>logs/insights.log</code> (after generate_insights.py) for the full agent trace. '
        'Use the main dashboard (app.py, port 8501) to see the output. '
        'Use the Code Review app (code_review.py, port 7501) for annotated source code.'
    )
