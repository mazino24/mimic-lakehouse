"""Shared DAG wiring: defaults, paths, callbacks, datasets."""

from __future__ import annotations

import os
from datetime import timedelta

import pendulum

from airflow.datasets import Dataset

SPARK_HOME_JOBS = "/opt/spark-apps/jobs"
DBT_PROJECT_DIR = "/opt/dbt"
POSTGRES_JAR = "/opt/spark-jars/postgresql.jar"

LOCAL_TZ = pendulum.timezone("UTC")

#: Datasets let the ML DAG start the moment the mart is refreshed, instead of
#: guessing at a cron offset that drifts as the ELT gets slower.
#: URIs follow AIP-60 (host/database/schema/table), which Airflow 3 enforces.
FEATURE_MART = Dataset("postgres://warehouse:5432/mimic/lake/feature_mart")
DQ_RESULTS = Dataset("postgres://warehouse:5432/mimic/lake/dq_results")

DEFAULT_ARGS = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    # Transient S3/JDBC hiccups resolve fast; a wedged Spark job does not.
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
    "execution_timeout": timedelta(hours=3),
}

SPARK_CONF = {
    "spark.executor.memory": os.getenv("SPARK_EXECUTOR_MEMORY", "2g"),
    "spark.executor.cores": os.getenv("SPARK_EXECUTOR_CORES", "2"),
    "spark.driver.memory": os.getenv("SPARK_DRIVER_MEMORY", "1g"),
    "spark.sql.shuffle.partitions": os.getenv("SPARK_SHUFFLE_PARTITIONS", "16"),
    "spark.hadoop.fs.s3a.endpoint": os.getenv("S3_ENDPOINT", "http://minio:9000"),
    "spark.hadoop.fs.s3a.access.key": os.getenv("S3_ACCESS_KEY", "minioadmin"),
    "spark.hadoop.fs.s3a.secret.key": os.getenv("S3_SECRET_KEY", "minioadmin"),
    "spark.hadoop.fs.s3a.path.style.access": "true",
    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
    "spark.hadoop.fs.s3a.aws.credentials.provider":
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    # The repo is bind-mounted into every Spark container, so executors import
    # `lakehouse` from the same source tree the driver does — no egg/zip to
    # rebuild on each edit.
    "spark.executorEnv.PYTHONPATH": "/opt/spark-apps",
}

SPARK_ENV = {
    "LAKE_ROOT": os.getenv("LAKE_ROOT", "s3a://mimic-lake"),
    "S3_ENDPOINT": os.getenv("S3_ENDPOINT", "http://minio:9000"),
    "S3_ACCESS_KEY": os.getenv("S3_ACCESS_KEY", "minioadmin"),
    "S3_SECRET_KEY": os.getenv("S3_SECRET_KEY", "minioadmin"),
    "WAREHOUSE_HOST": os.getenv("WAREHOUSE_HOST", "warehouse"),
    "WAREHOUSE_DB": os.getenv("WAREHOUSE_DB", "mimic"),
    "WAREHOUSE_USER": os.getenv("WAREHOUSE_USER", "mimic"),
    "WAREHOUSE_PASSWORD": os.getenv("WAREHOUSE_PASSWORD", "mimic"),
    "PYTHONPATH": "/opt/spark-apps",
}


def spark_task(task_id: str, job: str, *, args: list[str] | None = None, **kwargs):
    """A ``spark-submit`` task pointed at one job in ``spark/jobs``."""
    from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

    return SparkSubmitOperator(
        task_id=task_id,
        application=f"{SPARK_HOME_JOBS}/{job}",
        conn_id="spark_default",
        name=f"mimic-{task_id}",
        application_args=args or [],
        conf=SPARK_CONF,
        env_vars={**SPARK_ENV, "RUN_ID": "{{ ds }}"},
        jars=POSTGRES_JAR,
        verbose=False,
        **kwargs,
    )


def dbt_command(task_id: str, command: str, **kwargs):
    from airflow.operators.bash import BashOperator

    return BashOperator(
        task_id=task_id,
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            f"dbt {command} --profiles-dir {DBT_PROJECT_DIR} --target prod"
        ),
        **kwargs,
    )
