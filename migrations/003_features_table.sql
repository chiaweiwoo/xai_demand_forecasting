-- Precomputed feature store (written once by build_features.py)
-- Replaces runtime compute_features() calls in backtest and smoke_test.
-- Stores all 19 features + y for every (unique_id, week) combination.
CREATE TABLE IF NOT EXISTS features (
    unique_id        TEXT    NOT NULL,
    week             TEXT    NOT NULL,
    y                REAL,
    lag_1            REAL,
    lag_2            REAL,
    lag_4            REAL,
    lag_8            REAL,
    lag_52           REAL,
    rolling_4_mean   REAL,
    rolling_8_mean   REAL,
    rolling_13_mean  REAL,
    rolling_4_std    REAL,
    week_of_year     INTEGER,
    month            INTEGER,
    year             INTEGER,
    snap             INTEGER,
    has_event        INTEGER,
    event_type_enc   INTEGER,
    sell_price       REAL,
    price_change_pct REAL,
    dept_mean_sales  REAL,
    cat_mean_sales   REAL,
    PRIMARY KEY (unique_id, week)
);
CREATE INDEX IF NOT EXISTS idx_feat_week ON features(week);
