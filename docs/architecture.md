# Architecture

## Layer contracts

Each layer has one job and one guarantee. The guarantee is what downstream code
is allowed to assume without re-checking.

| Layer | Storage | Grain | Guarantee | Rebuilt by |
| --- | --- | --- | --- | --- |
| `raw` | MinIO / S3, CSV | source | byte-identical to what the source system sent | the extract process |
| `bronze` | Parquet | source | typed, lineage-stamped, otherwise unmodified | `bronze_ingest.py` |
| `silver` | Parquet | one row per entity | deduplicated, conformed, clinically annotated | `silver_transform.py` |
| `gold` | Parquet | analysis grain | labelled, filtered to the study cohort, feature-ready | `gold_build.py` |
| `lake.*` | Postgres | = gold | queryable by SQL clients and dbt | `publish_warehouse.py` |
| `staging` / `intermediate` | Postgres views | = gold | renamed, lightly derived | dbt |
| `marts` | Postgres tables | dimensional | tested, documented, indexed | dbt |
| `analytics` | Postgres tables | reporting | aggregate views for humans | dbt |

**Nothing skips a layer.** A model never reads silver; a dashboard never reads
gold Parquet. That is what makes it possible to change how labs are aggregated
without hunting through five notebooks for who else depended on the old shape.

## Why silver exists separately from gold

Silver is study-agnostic. `silver.labevents` is every in-window target lab for
every admission in the hospital, not only the angina cohort. A second study —
heart failure, sepsis, readmission — reuses silver untouched and writes its own
gold cohort. Folding the cohort filter into silver would save one job and make
the layer single-use.

## The heavy join

`labevents.csv` is ~40 GB and 130M+ rows. Naively joining it to admissions and
the lab dictionary is where a pandas pipeline dies. What makes it tractable:

1. **Explicit schema** — no inference pass over 40 GB.
2. **Column pruning before the join** — Parquet reads only what is projected.
3. **Broadcast the dictionary** — `d_labitems` is a few thousand rows; broadcast
   turns a shuffle join into a map-side join.
4. **Filter before widening** — the target-lab filter cuts the row count by
   ~an order of magnitude before the admission-window join runs.
5. **AQE with skew-join handling** — lab volume per admission is heavily
   skewed (an ICU stay can have thousands of draws, a day case has three).

## Partitioning

| Table | Partition key | Why |
| --- | --- | --- |
| `silver.admissions` | `admit_year` | year-scoped queries are the common access pattern |
| `silver.labevents` | `marker_type` | acute vs routine splits cleanly and is always filtered on |
| `gold.cohort` | `split` | training reads one split at a time |
| `gold.feature_mart` | `split` | same |
| `_quality.dq_results` | `layer`, `table_name` | monitoring queries one table's history |

Partition keys are **strings, never booleans**. Hive-style partition values
round-trip as text, so a boolean partition column comes back as `StringType`
and every downstream `when(col(...))` breaks. Learned that during this build;
`test_pipeline_end_to_end.py` now asserts the round-trip type.

## Orchestration

Three DAGs:

**`mimic_lakehouse_elt`** (daily 02:00) — the build. Task per stage, so a gold
failure re-runs gold and not the 40 GB ingest. Bronze splits reference tables
from event tables because they fail for different reasons and take different
amounts of time. Exposes the mart as an Airflow **Dataset** on success.

**`mimic_ml_training`** (dataset-triggered) — starts when the mart is actually
refreshed rather than on a cron offset that drifts as the ELT gets slower.
Trains, writes metrics to the warehouse, then runs a regression gate.

**`mimic_quality_monitor`** (dataset-triggered) — watches the *soft* signals:
warn-severity checks, feature null-rate drift, cohort size moving overnight.
Healthcare pipelines rarely crash; they quietly go 90 % null.

### Retry policy
Two retries, exponential backoff, 15-minute cap, 3-hour task timeout.
Transient S3/JDBC failures clear on the first retry; a wedged Spark job should
surface fast rather than occupy a slot all night.

## Scaling to the real extract

| Knob | Demo | 41 GB extract |
| --- | --- | --- |
| `SPARK_WORKERS` | 2 | 4–8 |
| `SPARK_WORKER_MEMORY` | 4G | 16–32G |
| `SPARK_SHUFFLE_PARTITIONS` | 16 | ~3× total executor cores (200–400) |
| bronze `--repartition` | default | 64–256 for `labevents` |

The code does not change. That is the point of keeping every path, endpoint and
credential in `lakehouse/config.py` and reading them from the environment.

## What would change in production

- **Delta Lake or Iceberg** instead of bare Parquet — ACID, time travel, schema
  evolution, `MERGE` for late-arriving records.
- **Incremental loads** — `labevents` grows continuously; full refresh is fine
  at 41 GB, wasteful at 400 GB. The lineage columns are already there to
  support it.
- **A real secret store** — credentials come from the environment today, which
  is correct for a demo and unacceptable for PHI in production.
- **Managed orchestration** — MWAA / Cloud Composer rather than a Compose file.
- **Column-level lineage** — OpenLineage emits from Spark and Airflow with
  little effort and answers "what breaks if I drop this column".
