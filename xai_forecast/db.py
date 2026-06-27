import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path('db/forecasting.db')
_MIGRATIONS_DIR = Path(__file__).parent.parent / 'migrations'


def _setup_schema(conn: sqlite3.Connection) -> None:
    """Apply all migrations/*.sql in sorted order. All statements use IF NOT EXISTS."""
    for path in sorted(_MIGRATIONS_DIR.glob('*.sql')):
        conn.executescript(path.read_text())


def _ensure_external_cols(conn: sqlite3.Connection) -> None:
    """Add external signal columns to the features table if not already present.
    Called by build_features.py before writing. Safe to call multiple times.
    """
    existing = {row[1] for row in conn.execute('PRAGMA table_info(features)').fetchall()}
    additions = [
        ('temp_mean',          'REAL'),
        ('temp_max',           'REAL'),
        ('temp_min',           'REAL'),
        ('precip',             'REAL'),
        ('heat_days',          'INTEGER'),
        ('gas_price',          'REAL'),
        ('consumer_sentiment', 'REAL'),
    ]
    for col, dtype in additions:
        if col not in existing:
            conn.execute(f'ALTER TABLE features ADD COLUMN {col} {dtype}')
    conn.commit()


def get_conn(path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    _setup_schema(conn)
    _ensure_external_cols(conn)
    return conn


# ── Raw data reads (used by backtest at each iteration) ──────────────────────

def load_raw_window(conn: sqlite3.Connection, week_start: str, week_end: str) -> pd.DataFrame:
    """
    Join weekly_sales + calendar + prices + item_meta for weeks (week_start, week_end].
    Used by backtest to build the feature matrix for one training window.
    week_start should be buffer_start (window_start - 52 weeks) so lag_52 is correct.
    """
    return pd.read_sql(
        """
        SELECT ws.week, ws.unique_id, ws.y,
               c.snap, c.has_event, c.event_type_enc,
               p.sell_price,
               m.dept_mean_sales, m.cat_mean_sales
        FROM weekly_sales ws
        LEFT JOIN calendar  c ON c.week      = ws.week
        LEFT JOIN prices    p ON p.week      = ws.week AND p.unique_id = ws.unique_id
        LEFT JOIN item_meta m ON m.unique_id = ws.unique_id
        WHERE ws.week > ? AND ws.week <= ?
        ORDER BY ws.unique_id, ws.week
        """,
        conn, params=(week_start, week_end),
    )


def get_all_weeks(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute('SELECT DISTINCT week FROM weekly_sales ORDER BY week')
    return [r[0] for r in cur.fetchall()]


# ── Feature store reads (used by backtest after build_features.py) ────────────

def load_features_window(conn: sqlite3.Connection, week_start: str, week_end: str) -> pd.DataFrame:
    """Training window: all precomputed feature rows for (week_start, week_end]."""
    return pd.read_sql(
        'SELECT * FROM features WHERE week > ? AND week <= ? ORDER BY unique_id, week',
        conn, params=(week_start, week_end),
    )


def load_features_week(conn: sqlite3.Connection, week: str) -> pd.DataFrame:
    """Single forecast week: one precomputed feature row per SKU."""
    return pd.read_sql(
        'SELECT * FROM features WHERE week = ?',
        conn, params=(week,),
    )


# ── Ingest writes ─────────────────────────────────────────────────────────────

def insert_raw(conn: sqlite3.Connection, weekly_sales: pd.DataFrame,
               calendar: pd.DataFrame, prices: pd.DataFrame,
               item_meta: pd.DataFrame) -> None:
    weekly_sales.to_sql('weekly_sales', conn, if_exists='append', index=False, chunksize=10_000)
    calendar.to_sql('calendar',         conn, if_exists='append', index=False, chunksize=10_000)
    prices.to_sql('prices',             conn, if_exists='append', index=False, chunksize=10_000)
    item_meta.to_sql('item_meta',       conn, if_exists='append', index=False, chunksize=10_000)
    conn.commit()



# ── Backtest writes ───────────────────────────────────────────────────────────

def insert_forecasts(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        'INSERT OR REPLACE INTO forecasts (week_id, item_id, h1, trained_at) '
        'VALUES (:week_id, :item_id, :h1, :trained_at)', rows,
    )
    conn.commit()


def insert_evaluations(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        'INSERT OR REPLACE INTO evaluations '
        '(week_id, item_id, h1_mape, h1_mae, is_bad_week, mape_zscore) '
        'VALUES (:week_id, :item_id, :h1_mape, :h1_mae, :is_bad_week, :mape_zscore)', rows,
    )
    conn.commit()


def insert_xai(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        'INSERT OR REPLACE INTO xai_results (week_id, item_id, xai_type, payload) '
        'VALUES (:week_id, :item_id, :xai_type, :payload)', rows,
    )
    conn.commit()


# ── Dashboard reads ───────────────────────────────────────────────────────────

def load_evaluations(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql('SELECT * FROM evaluations ORDER BY week_id', conn)


def load_xai(conn: sqlite3.Connection, week_id: str, item_id: str | None = None) -> list[dict]:
    if item_id:
        cur = conn.execute(
            'SELECT * FROM xai_results WHERE week_id=? AND item_id=?', (week_id, item_id)
        )
    else:
        cur = conn.execute('SELECT * FROM xai_results WHERE week_id=?', (week_id,))
    return [dict(r) for r in cur.fetchall()]


def load_all_shap_payloads(conn: sqlite3.Connection) -> list[dict]:
    """All SHAP payloads for bad weeks (for recurring-drivers aggregation)."""
    cur = conn.execute(
        'SELECT week_id, item_id, payload FROM xai_results WHERE xai_type=?', ('shap',)
    )
    return [dict(r) for r in cur.fetchall()]


# ── Insight reads/writes ──────────────────────────────────────────────────────

def insert_insight_finding(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        'INSERT OR REPLACE INTO insight_findings '
        '(finding_id, finding_type, status, confidence, evidence, hypothesis, critic_notes, created_at) '
        'VALUES (:finding_id, :finding_type, :status, :confidence, :evidence, :hypothesis, :critic_notes, :created_at)',
        row,
    )
    conn.commit()


def insert_insight_summary(conn: sqlite3.Connection, data_scientist: dict,
                            business_leader: dict, model_flash: str, model_critic: str) -> None:
    conn.execute(
        'INSERT OR REPLACE INTO insight_summary '
        '(key, data_scientist, business_leader, model_flash, model_critic, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        ('overall', json.dumps(data_scientist), json.dumps(business_leader),
         model_flash, model_critic, datetime.utcnow().isoformat()),
    )
    conn.commit()


def load_insight_summary(conn: sqlite3.Connection) -> dict | None:
    cur = conn.execute(
        'SELECT data_scientist, business_leader, model_flash, model_critic, created_at '
        'FROM insight_summary WHERE key=?', ('overall',)
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        'data_scientist':  json.loads(row[0]),
        'business_leader': json.loads(row[1]),
        'model_flash':     row[2],
        'model_critic':    row[3],
        'created_at':      row[4],
    }


def load_insight_findings(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        'SELECT finding_id, finding_type, status, confidence, evidence, hypothesis, critic_notes, created_at '
        'FROM insight_findings ORDER BY finding_id'
    )
    rows = []
    for r in cur.fetchall():
        rows.append({
            'finding_id':   r[0],
            'finding_type': r[1],
            'status':       r[2],
            'confidence':   r[3],
            'evidence':     json.loads(r[4]) if r[4] else {},
            'hypothesis':   json.loads(r[5]) if r[5] else None,
            'critic_notes': r[6],
            'created_at':   r[7],
        })
    return rows


def week_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        '''SELECT week_id,
                  COUNT(*)         AS n_items,
                  AVG(h1_mape)     AS avg_mape,
                  SUM(is_bad_week) AS n_bad_items,
                  AVG(mape_zscore) AS avg_zscore
           FROM evaluations
           GROUP BY week_id
           ORDER BY week_id''',
        conn,
    )
