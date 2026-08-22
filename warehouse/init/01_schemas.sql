-- Warehouse layout.
--
--   lake          <- written by Spark (publish_warehouse job) and the ML DAG
--   staging       <- dbt views, 1:1 with lake
--   intermediate  <- dbt ephemeral models
--   marts         <- dbt dimensional models + the ML contract table
--   analytics     <- dbt reporting models

CREATE SCHEMA IF NOT EXISTS lake;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS intermediate;
CREATE SCHEMA IF NOT EXISTS marts;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Metrics written back by the training DAG. Created up front so dbt's source
-- freshness and the analytics model resolve on a cold start, before the first
-- model has ever been trained.
CREATE TABLE IF NOT EXISTS lake.model_metrics (
    run_id        TEXT        NOT NULL,
    trained_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_name    TEXT        NOT NULL,
    split         TEXT        NOT NULL,
    metric_name   TEXT        NOT NULL,
    metric_value  DOUBLE PRECISION,
    n_rows        BIGINT,
    n_features    INTEGER,
    PRIMARY KEY (run_id, model_name, split, metric_name)
);

CREATE INDEX IF NOT EXISTS model_metrics_trained_at_idx
    ON lake.model_metrics (trained_at DESC);

COMMENT ON SCHEMA lake IS 'Gold-layer tables published from the Spark lakehouse';
COMMENT ON SCHEMA marts IS 'dbt dimensional models and the ML feature contract';
