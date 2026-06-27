"""Shared pytest fixtures."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from xai_forecast.db import get_conn
from xai_forecast.features import FEATURE_COLS, compute_features
from xai_forecast.train import train_model
from xai_forecast.xai import make_explainer


# ── Minimal raw DataFrame ────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Two SKUs over 60 weeks (enough for lag_52 on last 8 weeks)."""
    weeks = pd.date_range('2013-01-05', periods=60, freq='7D').strftime('%Y-%m-%d').tolist()
    rng = np.random.default_rng(0)
    records = []
    for uid in ['CA_1_001_TX_1', 'CA_1_002_TX_1']:
        for i, w in enumerate(weeks):
            records.append({
                'week': w, 'unique_id': uid, 'y': float(rng.integers(0, 10)),
                'snap': int(rng.integers(0, 2)), 'has_event': 0, 'event_type_enc': 0,
                'sell_price': 2.0 + 0.1 * (i % 5),
                'dept_mean_sales': 5.0, 'cat_mean_sales': 10.0,
            })
    return pd.DataFrame(records)


# ── Trained model + explainer ────────────────────────────────────────────────

@pytest.fixture(scope='module')
def trained_model_and_explainer():
    """Small but real LightGBM model trained on synthetic feature rows."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    df['y'] = np.abs(rng.standard_normal(n)) * 5 + 1
    df['unique_id'] = 'test_item'
    model = train_model(df)
    explainer = make_explainer(model)
    return model, explainer


# ── In-process SQLite DB ─────────────────────────────────────────────────────

@pytest.fixture
def db_conn():
    """Fresh SQLite DB in a temp file with full schema applied."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = get_conn(path)
    yield conn
    conn.close()
    os.unlink(path)
