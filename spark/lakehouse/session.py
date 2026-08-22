"""SparkSession factory shared by every job.

The same builder is used by the Docker Spark cluster and by the unit tests;
S3A wiring is only attached when ``S3_ENABLED`` is on, so tests run against the
local filesystem with no MinIO in sight.
"""

from __future__ import annotations

import logging
import os

from pyspark.sql import SparkSession

from lakehouse.config import S3, S3Config

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> logging.Logger:
    logging.basicConfig(
        level=(level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        format=LOG_FORMAT,
        force=True,
    )
    return logging.getLogger("lakehouse")


def build_session(
    app_name: str,
    *,
    s3: S3Config | None = None,
    shuffle_partitions: int | None = None,
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    """Create (or fetch) the SparkSession for a job."""
    s3 = s3 or S3
    builder = (
        SparkSession.builder.appName(app_name)
        # Adaptive execution keeps the 40 GB labevents shuffle from exploding
        # into thousands of tiny partitions on the small demo cluster.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # STATIC (the default) is deliberate: these jobs are full refreshes,
        # so an overwrite must clear partitions that no longer exist. Under
        # DYNAMIC, a renamed partition key silently leaves orphaned
        # directories behind and the next read fails on conflicting schemas.
        .config("spark.sql.sources.partitionOverwriteMode", "static")
        .config("spark.sql.session.timeZone", "UTC")
    )

    partitions = shuffle_partitions or int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "0") or 0)
    if partitions:
        builder = builder.config("spark.sql.shuffle.partitions", str(partitions))

    if s3.enabled:
        builder = (
            builder.config("spark.hadoop.fs.s3a.endpoint", s3.endpoint)
            .config("spark.hadoop.fs.s3a.access.key", s3.access_key)
            .config("spark.hadoop.fs.s3a.secret.key", s3.secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", str(s3.path_style_access).lower())
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled",
                    str(s3.endpoint.startswith("https")).lower())
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            )
        )

    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
    return session
