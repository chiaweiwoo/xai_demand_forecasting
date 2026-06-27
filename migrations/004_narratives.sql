CREATE TABLE IF NOT EXISTS narratives (
    scope      TEXT NOT NULL,   -- 'week' | 'item' | 'executive'
    key        TEXT NOT NULL,   -- week_id for week/executive; 'week_id::item_id' for item
    payload    TEXT NOT NULL,   -- JSON: {headline, body, primary_driver, confidence, model}
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (scope, key)
);
