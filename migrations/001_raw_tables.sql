-- Raw ingestion tables (written once by ingest.py)
-- week = Saturday date string (Walmart fiscal week start, e.g. '2011-01-29')
CREATE TABLE IF NOT EXISTS weekly_sales (
    week      TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    y         REAL,
    PRIMARY KEY (week, unique_id)
);
CREATE INDEX IF NOT EXISTS idx_ws_week ON weekly_sales(week);
CREATE INDEX IF NOT EXISTS idx_ws_uid  ON weekly_sales(unique_id);

CREATE TABLE IF NOT EXISTS calendar (
    week           TEXT PRIMARY KEY,
    snap           INTEGER,
    has_event      INTEGER,
    event_type_enc INTEGER
);

CREATE TABLE IF NOT EXISTS prices (
    week       TEXT NOT NULL,
    unique_id  TEXT NOT NULL,
    sell_price REAL,
    PRIMARY KEY (week, unique_id)
);
CREATE INDEX IF NOT EXISTS idx_prices_week ON prices(week);
CREATE INDEX IF NOT EXISTS idx_prices_uid  ON prices(unique_id);

CREATE TABLE IF NOT EXISTS item_meta (
    unique_id        TEXT PRIMARY KEY,
    dept_id          TEXT,
    cat_id           TEXT,
    dept_mean_sales  REAL,
    cat_mean_sales   REAL
);
