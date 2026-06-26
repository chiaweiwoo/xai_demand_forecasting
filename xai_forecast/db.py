import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path('db/forecasting.db')


def get_conn(path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn


def init_db(path: str | Path = DB_PATH) -> None:
    """Run all pending migrations to initialise or upgrade the DB."""
    import migrate
    migrate.run(str(path))


# ── Ingest writes ─────────────────────────────────────────────────────────────

def insert_raw(conn: sqlite3.Connection, weekly_sales: pd.DataFrame,
               calendar: pd.DataFrame, prices: pd.DataFrame,
               item_meta: pd.DataFrame) -> None:
    weekly_sales.to_sql('weekly_sales', conn, if_exists='append', index=False, chunksize=10_000)
    calendar.to_sql('calendar',         conn, if_exists='append', index=False, chunksize=10_000)
    prices.to_sql('prices',             conn, if_exists='append', index=False, chunksize=10_000)
    item_meta.to_sql('item_meta',       conn, if_exists='append', index=False, chunksize=10_000)
    conn.commit()


# ── Feature writes / reads ────────────────────────────────────────────────────

def insert_features(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    df.to_sql('features', conn, if_exists='append', index=False, chunksize=10_000)
    conn.commit()


def get_weeks(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute('SELECT DISTINCT week FROM features ORDER BY week')
    return [r[0] for r in cur.fetchall()]


def load_features_window(conn: sqlite3.Connection, week_start: str, week_end: str) -> pd.DataFrame:
    return pd.read_sql(
        'SELECT * FROM features WHERE week > ? AND week <= ?',
        conn, params=(week_start, week_end),
    )


def load_features_week(conn: sqlite3.Connection, week: str) -> pd.DataFrame:
    return pd.read_sql(
        'SELECT * FROM features WHERE week = ?',
        conn, params=(week,),
    )


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
