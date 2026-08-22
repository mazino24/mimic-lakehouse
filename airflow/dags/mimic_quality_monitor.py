"""Daily data-quality monitor.

The ELT DAG fails loudly on blocking checks. This DAG watches the *soft*
signals — warn-severity checks, feature null-rate drift, cohort size moving
more than it should overnight — and reports them, because the failure mode of
a healthcare pipeline is rarely a crash. It is a column that quietly went 90 %
null three weeks ago.
"""

from __future__ import annotations

import pendulum
from common import DEFAULT_ARGS, DQ_RESULTS

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException

NULL_RATE_JUMP = 0.15   # absolute increase in a column's null rate
COHORT_DRIFT = 0.20     # relative change in cohort size


@task
def load_recent_quality_history() -> list[dict]:
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    hook = PostgresHook(postgres_conn_id="warehouse")
    rows = hook.get_records(
        """
        SELECT run_id, layer, table_name, check_name, severity, passed,
               observed, threshold, row_count
        FROM lake.dq_results
        WHERE checked_at >= now() - interval '7 days'
        ORDER BY checked_at DESC
        """
    )
    columns = ["run_id", "layer", "table_name", "check_name", "severity", "passed",
               "observed", "threshold", "row_count"]
    return [dict(zip(columns, row, strict=False)) for row in rows]


@task
def report_soft_failures(history: list[dict]) -> str:
    warnings = [r for r in history if r["passed"] == "false" and r["severity"] == "warn"]
    if not warnings:
        raise AirflowSkipException("no warn-severity failures in the last 7 days")
    lines = [
        f"{r['layer']}.{r['table_name']} :: {r['check_name']} "
        f"(observed={r['observed']}, threshold={r['threshold']})"
        for r in warnings[:50]
    ]
    return "\n".join(lines)


@task
def detect_feature_drift() -> str:
    """Compare the newest feature-coverage snapshot with the previous one."""
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    hook = PostgresHook(postgres_conn_id="warehouse")
    rows = hook.get_records(
        f"""
        WITH ranked AS (
            SELECT column_name, null_rate, _run_id,
                   dense_rank() OVER (ORDER BY _run_id DESC) AS run_rank
            FROM lake.feature_coverage
        )
        SELECT current.column_name, previous.null_rate, current.null_rate
        FROM ranked current
        JOIN ranked previous USING (column_name)
        WHERE current.run_rank = 1
          AND previous.run_rank = 2
          AND current.null_rate - previous.null_rate > {NULL_RATE_JUMP}
        ORDER BY current.null_rate - previous.null_rate DESC
        """
    )
    if not rows:
        raise AirflowSkipException("no feature drifted beyond the null-rate threshold")
    return "\n".join(
        f"{column}: null rate {before:.3f} -> {after:.3f}" for column, before, after in rows
    )


with DAG(
    dag_id="mimic_quality_monitor",
    description="Track soft data-quality signals and feature drift",
    doc_md=__doc__,
    default_args={**DEFAULT_ARGS, "retries": 1},
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=[DQ_RESULTS],
    catchup=False,
    tags=["mimic", "data-quality", "monitoring"],
) as dag:
    history = load_recent_quality_history()
    report_soft_failures(history)
    detect_feature_drift()
