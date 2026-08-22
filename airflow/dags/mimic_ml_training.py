"""Training DAG — the downstream consumer of the warehouse.

Dataset-scheduled: it wakes when ``mimic_lakehouse_elt`` publishes a new
feature mart, so there is no cron offset to keep in sync. Metrics land back in
the warehouse (``analytics.model_metrics``), which makes model performance a
queryable time series next to the data-quality history that explains it.
"""

from __future__ import annotations

import pendulum
from common import DEFAULT_ARGS, FEATURE_MART

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

with DAG(
    dag_id="mimic_ml_training",
    description="Train and evaluate angina models on the published feature mart",
    doc_md=__doc__,
    default_args=DEFAULT_ARGS,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=[FEATURE_MART],  # dataset-triggered
    catchup=False,
    max_active_runs=1,
    tags=["mimic", "ml", "downstream"],
) as dag:
    start = EmptyOperator(task_id="start")

    train = BashOperator(
        task_id="train_and_evaluate",
        bash_command=(
            "python /opt/ml/train_angina_model.py "
            "--source warehouse "
            "--run-id {{ ds }} "
            "--output-dir /opt/ml/artifacts/{{ ds }}"
        ),
    )

    # A model that got worse is a data problem until proven otherwise, so the
    # gate lives in the pipeline rather than in someone's notebook.
    gate = BashOperator(
        task_id="check_model_regression",
        bash_command=(
            "python /opt/ml/check_regression.py "
            "--run-id {{ ds }} --metric roc_auc --max-drop 0.03"
        ),
    )

    finish = EmptyOperator(task_id="finish")

    start >> train >> gate >> finish
