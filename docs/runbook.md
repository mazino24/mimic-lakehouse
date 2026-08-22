# Runbook

## Daily operation

The ELT DAG runs at 02:00 UTC. Training and monitoring are dataset-triggered,
so they follow automatically when the mart is refreshed.

Healthy morning check:

```sql
-- Did last night's run pass every gate?
select * from analytics.analytics_data_quality_history
order by checked_at desc limit 20;

-- Did anything drift?
select * from analytics.analytics_feature_coverage
where completeness_tier in ('sparse', 'mostly_missing')
order by null_rate desc;

-- Is the model still where it was?
select * from analytics.analytics_model_performance
order by trained_at desc limit 10;
```

## Failure playbooks

### `DataQualityError` in a Spark task
The gate did its job — data was about to enter a layer in a state that
invalidates everything downstream.

1. Read the task log; the message names the check and the observed value.
2. Query the history for context:
   ```sql
   select * from analytics.analytics_data_quality_history
   where table_name = '<table>' order by checked_at desc limit 10;
   ```
3. If the source extract is at fault, fix the extract and clear the task.
4. If the check itself is wrong, change it in `spark/jobs/*.py` **with a test**.
   Never lower a threshold to make a red run green — write down why the new
   threshold is correct.

### `dbt test` failure
Data landed in the warehouse but broke a modelling assumption. The marts from
the previous run are still there — dbt failed *after* building, so consumers
see stale-but-valid data rather than fresh-but-wrong data.

```bash
make shell-airflow
cd /opt/dbt && dbt test --profiles-dir . --target prod --store-failures
# then inspect the failure table dbt created
```

### Spark job OOM
Usually the `labevents` join. In order of preference:

1. Raise `SPARK_SHUFFLE_PARTITIONS` (smaller partitions, less memory each).
2. Raise `SPARK_WORKER_MEMORY` / `SPARK_EXECUTOR_MEMORY`.
3. Check for a skewed partition in the Spark UI — AQE skew handling is on, but
   an extreme outlier stay can still dominate.
4. Ingest `labevents` with a higher `--repartition` so downstream reads
   parallelise better.

### The model regressed
`check_regression.py` failed. **Suspect the data first** — the model code did
not change; the data did.

```sql
select * from analytics.analytics_feature_coverage order by null_rate desc limit 20;
select * from analytics.analytics_cohort_summary;
select * from analytics.analytics_data_quality_history
where not run_healthy order by checked_at desc;
```

A feature that jumped from 20 % to 90 % null explains a ROC-AUC drop far more
often than anything in the training script.

### Airflow shows no DAGs
```bash
make logs                      # scheduler import errors appear here
docker compose exec airflow-scheduler airflow dags list-import-errors
```

## Common tasks

```bash
make demo                # everything, from an empty machine
make test                # test suite, no Docker
make dbt-run dbt-test    # rebuild and re-test the warehouse models
make dbt-docs            # dbt docs site on :8088
make psql                # warehouse shell
make clean               # stop and delete all volumes
```

### Backfilling a date range
```bash
docker compose exec airflow-scheduler \
  airflow dags backfill mimic_lakehouse_elt -s 2026-08-01 -e 2026-08-07
```
The pipeline is a full refresh, so a backfill re-derives the same tables; the
value is in re-stamping `_run_id` and re-recording quality history.

### Rebuilding one layer
```bash
docker compose exec airflow-scheduler bash -c \
  "cd /opt/spark-apps && spark-submit --master spark://spark-master:7077 jobs/gold_build.py"
```

### Rotating credentials
Edit `.env`, then `make restart`. Nothing reads a credential from source —
`lakehouse/config.py` is the only place they enter the code, and it reads the
environment.
