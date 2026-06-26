-- Backtest output tables (written by backtest.py)
CREATE TABLE IF NOT EXISTS forecasts (
    week_id    TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    h1         REAL,
    trained_at TEXT,
    PRIMARY KEY (week_id, item_id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    week_id     TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    h1_mape     REAL,
    h1_mae      REAL,
    is_bad_week INTEGER DEFAULT 0,
    mape_zscore REAL,
    PRIMARY KEY (week_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_eval_week ON evaluations(week_id);

CREATE TABLE IF NOT EXISTS xai_results (
    week_id  TEXT NOT NULL,
    item_id  TEXT NOT NULL,
    xai_type TEXT NOT NULL,
    payload  TEXT,
    PRIMARY KEY (week_id, item_id, xai_type)
);
CREATE INDEX IF NOT EXISTS idx_xai_week ON xai_results(week_id);
CREATE INDEX IF NOT EXISTS idx_xai_item ON xai_results(item_id);
