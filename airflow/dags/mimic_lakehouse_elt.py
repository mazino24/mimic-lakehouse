"""Main ELT DAG: raw MIMIC-IV CSVs -> bronze -> silver -> gold -> warehouse -> dbt.

Design notes
------------
* Every Spark stage is its own task, so a failure in ``gold`` re-runs ``gold``
  and not the 40 GB bronze ingest.
* Data-quality gates run *inside* the Spark jobs and raise, which fails the
  task; the ``dbt test`` task is the second gate, at the warehouse boundary.
* The mart is exposed as an Airflow Dataset, so the training DAG is triggered
  by data landing rather than by a cron guess.
"""

from __future__ import annotations

import pendulum
from common import DEFAULT_ARGS, DQ_RESULTS, FEATURE_MART, dbt_command, spark_task

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

DOC = __doc__

with DAG(
    dag_id="mimic_lakehouse_elt",
    description="Bronze/silver/gold lakehouse build for the MIMIC-IV angina study",
    doc_md=DOC,
    default_args=DEFAULT_ARGS,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mimic", "lakehouse", "spark", "dbt"],
) as dag:
    start = EmptyOperator(task_id="start")

    with TaskGroup(group_id="bronze", tooltip="Typed Parquet copy of the raw extracts") as bronze:
        ingest_reference = spark_task(
            "ingest_reference_tables",
            "bronze_ingest.py",
            args=["--tables", "patients", "admissions", "diagnoses_icd", "d_labitems",
                  "ecg_record_list", "--skip-missing"],
        )
        # labevents is ~40 GB on its own: separate task, its own retries, and
        # enough output files that downstream reads parallelise properly.
        ingest_events = spark_task(
            "ingest_event_tables",
            "bronze_ingest.py",
            args=["--tables", "labevents", "chartevents", "--repartition", "64",
                  "--skip-missing"],
        )
        # The two ingests are independent; they fan out inside the group.

    silver = spark_task("silver_transform", "silver_transform.py")

    gold = spark_task("gold_build", "gold_build.py", args=["--min-coverage", "0.02"])

    publish = spark_task("publish_to_warehouse", "publish_warehouse.py",
                         outlets=[FEATURE_MART, DQ_RESULTS])

    with TaskGroup(group_id="dbt", tooltip="SQL modelling and warehouse-level tests") as dbt:
        dbt_deps = dbt_command("deps", "deps")
        dbt_run = dbt_command("run", "run")
        dbt_test = dbt_command("test", "test")
        dbt_docs = dbt_command("docs_generate", "docs generate")
        dbt_deps >> dbt_run >> dbt_test >> dbt_docs

    finish = EmptyOperator(task_id="finish")

    start >> bronze >> silver >> gold >> publish >> dbt >> finish
