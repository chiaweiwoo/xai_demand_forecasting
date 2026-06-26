import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path('db/forecasting.db')

_DDL = """
CREATE TABLE IF NOT EXISTS features (
    week        TEXT NOT NULL,
    unique_id   TEXT NOT NULL,
    y           REAL,
    lag_1 REAL, lag_2 REAL, lag_4 REAL, lag_8 REAL, lag_52 REAL,
    rolling_4_mean REAL, rolling_8_mean REAL, rolling_13_mean REAL, rolling_4_std REAL,
    week_of_year INTEGER, month INTEGER, year INTEGER,
    snap INTEGER, has_event INTEGER, event_type_enc INTEGER,
    sell_price REAL, price_change_pct REAL,
    dept_enc INTEGER, cat_enc INTEGER,
    PRIMARY KEY (week, unique_id)
);
CREATE INDEX IF NOT EXISTS idx_features_week    ON features(week);
CREATE INDEX IF NOT EXISTS idx_features_uid     ON features(unique_id);
CREATE TABLE IF NOT EXISTS forecasts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id     TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    h1          REAL,
    trained_at  TEXT
);
CREATE TABLE IF NOT EXISTS actuals (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    sales   REAL,
    UNIQUE(week_id, item_id)
);
CREATE TABLE IF NOT EXISTS evaluations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id     TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    h1_mape     REAL,
    h1_mae      REAL,
    is_bad_week INTEGER DEFAULT 0,
    mape_zscore REAL
);
CREATE TABLE IF NOT EXISTS xai_results (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id  TEXT NOT NULL,
    item_id  TEXT NOT NULL,
    xai_type TEXT NOT NULL,
    payload  TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_week ON evaluations(week_id);
CREATE INDEX IF NOT EXISTS idx_xai_week  ON xai_results(week_id);
CREATE INDEX IF NOT EXISTS idx_xai_item  ON xai_results(item_id);
"""


def get_conn(path: str | Path = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | Path = DB_PATH) -> None:
    with get_conn(path) as conn:
        conn.executescript(_DDL)


def insert_features(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    df.to_sql('features', conn, if_exists='append', index=False, chunksize=10_000)
    conn.commit()


def load_features_window(conn: sqlite3.Connection, week_start: str, week_end: str) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT * FROM features WHERE week > ? AND week <= ?",
        conn, params=(week_start, week_end),
    )


def load_features_week(conn: sqlite3.Connection, week: str) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT * FROM features WHERE week = ?",
        conn, params=(week,),
    )


def get_weeks(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute("SELECT DISTINCT week FROM features ORDER BY week")
    return [r[0] for r in cur.fetchall()]


def insert_forecasts(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        "INSERT INTO forecasts (week_id, item_id, h1, trained_at) "
        "VALUES (:week_id, :item_id, :h1, :trained_at)",
        rows,
    )
    conn.commit()


def insert_actuals(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO actuals (week_id, item_id, sales) "
        "VALUES (:week_id, :item_id, :sales)",
        rows,
    )
    conn.commit()


def insert_evaluations(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        "INSERT INTO evaluations (week_id, item_id, h1_mape, h1_mae, is_bad_week, mape_zscore) "
        "VALUES (:week_id, :item_id, :h1_mape, :h1_mae, :is_bad_week, :mape_zscore)",
        rows,
    )
    conn.commit()


def insert_xai(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        "INSERT INTO xai_results (week_id, item_id, xai_type, payload) "
        "VALUES (:week_id, :item_id, :xai_type, :payload)",
        rows,
    )
    conn.commit()


def load_evaluations(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM evaluations ORDER BY week_id", conn)


def load_xai(conn: sqlite3.Connection, week_id: str, item_id: str | None = None) -> list[dict]:
    if item_id:
        cur = conn.execute(
            "SELECT * FROM xai_results WHERE week_id=? AND item_id=?", (week_id, item_id)
        )
    else:
        cur = conn.execute("SELECT * FROM xai_results WHERE week_id=?", (week_id,))
    return [dict(r) for r in cur.fetchall()]


def week_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT week_id,
               COUNT(*)        AS n_items,
               AVG(h1_mape)    AS avg_mape,
               SUM(is_bad_week) AS n_bad_items,
               AVG(mape_zscore) AS avg_zscore
        FROM evaluations
        GROUP BY week_id
        ORDER BY week_id
        """,
        conn,
    )
