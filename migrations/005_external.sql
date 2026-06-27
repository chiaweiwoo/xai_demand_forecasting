CREATE TABLE IF NOT EXISTS external_signals (
    week                TEXT PRIMARY KEY,  -- Saturday fiscal-week date, joins to features.week
    temp_mean           REAL,              -- weekly mean of daily mean temp (C)
    temp_max            REAL,              -- weekly max of daily max temp (C)
    temp_min            REAL,              -- weekly min of daily min temp (C)
    precip              REAL,              -- weekly total precipitation (mm)
    heat_days           INTEGER,           -- days in week with daily max > 32C (~90F)
    gas_price           REAL,              -- CA regular retail gasoline ($/gal)
    consumer_sentiment  REAL               -- U. Michigan consumer sentiment index
);

CREATE INDEX IF NOT EXISTS idx_external_week ON external_signals (week);
