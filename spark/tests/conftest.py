"""Shared pytest fixtures: one local SparkSession for the whole test session."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tests never touch MinIO; they run against the local filesystem.
os.environ.setdefault("S3_ENABLED", "false")
os.environ.setdefault("RUN_ID", "pytest")
# Spark 3.5 on JDK 21+ needs the security manager explicitly allowed.
os.environ.setdefault("JDK_JAVA_OPTIONS", "-Djava.security.manager=allow")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark():
    from lakehouse.session import build_session

    session = build_session(
        "lakehouse-tests",
        shuffle_partitions=2,
        extra_conf={"spark.master": "local[2]", "spark.ui.enabled": "false"},
    )
    yield session
    session.stop()


def ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


@pytest.fixture
def make_df(spark):
    def _make(rows, schema):
        return spark.createDataFrame(rows, schema)

    return _make
