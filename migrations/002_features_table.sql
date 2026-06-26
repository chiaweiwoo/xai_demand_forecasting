-- Engineered feature matrix (written once by build_features.py)
CREATE TABLE IF NOT EXISTS features (
    week           TEXT NOT NULL,
    unique_id      TEXT NOT NULL,
    y              REAL,
    lag_1 REAL, lag_2 REAL, lag_4 REAL, lag_8 REAL, lag_52 REAL,
    rolling_4_mean REAL, rolling_8_mean REAL, rolling_13_mean REAL,
    rolling_4_std  REAL,
    week_of_year   INTEGER, month INTEGER, year INTEGER,
    snap           INTEGER, has_event INTEGER, event_type_enc INTEGER,
    sell_price     REAL, price_change_pct REAL,
    dept_enc       INTEGER, cat_enc INTEGER,
    PRIMARY KEY (week, unique_id)
);
CREATE INDEX IF NOT EXISTS idx_feat_week ON features(week);
CREATE INDEX IF NOT EXISTS idx_feat_uid  ON features(unique_id);
