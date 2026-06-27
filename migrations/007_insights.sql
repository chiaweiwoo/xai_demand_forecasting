-- Stage 3: insights module replaces the narrative layer.
-- Drops the narratives table and creates insight_findings + insight_summary.
-- Applied automatically by get_conn() via _setup_schema().

DROP TABLE IF EXISTS narratives;

CREATE TABLE IF NOT EXISTS insight_findings (
    finding_id   TEXT NOT NULL,
    finding_type TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',  -- accepted / rejected / needs_review
    confidence   TEXT NOT NULL DEFAULT 'low',      -- high / medium / low
    evidence     TEXT NOT NULL,                    -- JSON: raw evidence pack
    hypothesis   TEXT,                             -- JSON: LLM hypothesis
    critic_notes TEXT,                             -- JSON: critic response
    created_at   TEXT NOT NULL,
    PRIMARY KEY (finding_id)
);

CREATE TABLE IF NOT EXISTS insight_summary (
    key              TEXT NOT NULL,  -- always 'overall'
    data_scientist   TEXT NOT NULL,  -- JSON: DS-facing findings
    business_leader  TEXT NOT NULL,  -- JSON: business-facing summary
    model_flash      TEXT NOT NULL,
    model_critic     TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (key)
);

CREATE INDEX IF NOT EXISTS idx_insight_findings_type   ON insight_findings (finding_type);
CREATE INDEX IF NOT EXISTS idx_insight_findings_status ON insight_findings (status);
