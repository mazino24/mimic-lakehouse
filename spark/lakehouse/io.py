"""Read/write helpers that keep every layer consistent.

Each write stamps lineage columns (``_loaded_at``, ``_run_id``, ``_source``) so
any row in the warehouse can be traced back to the job execution that produced
it — the thing you always wish you had when a mart looks wrong at 3am.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from lakehouse.config import LAKE, WAREHOUSE, LakeConfig, WarehouseConfig

log = logging.getLogger("lakehouse.io")


def run_id() -> str:
    """Airflow's logical date when orchestrated, else a manual marker."""
    return os.getenv("RUN_ID") or os.getenv("AIRFLOW_CTX_LOGICAL_DATE") or "manual"


def with_lineage(df: DataFrame, source: str) -> DataFrame:
    return (
        df.withColumn("_loaded_at", F.current_timestamp())
        .withColumn("_run_id", F.lit(run_id()))
        .withColumn("_source", F.lit(source))
    )


def read_csv(
    spark: SparkSession,
    path: str,
    schema: StructType,
    *,
    timestamp_format: str = "yyyy-MM-dd HH:mm:ss",
) -> DataFrame:
    """Read a raw MIMIC CSV with an explicit schema.

    ``PERMISSIVE`` + ``_corrupt_record`` would double the columns downstream, so
    malformed rows are captured by counting nulls in the DQ layer instead.
    """
    return (
        spark.read.option("header", True)
        .option("timestampFormat", timestamp_format)
        .option("mode", "PERMISSIVE")
        .option("multiLine", False)
        .option("escape", '"')
        .schema(schema)
        .csv(path)
    )


def write_parquet(
    df: DataFrame,
    path: str,
    *,
    partition_by: Sequence[str] = (),
    mode: str = "overwrite",
    repartition: int | None = None,
) -> str:
    writer_df = df.repartition(repartition) if repartition else df
    writer = writer_df.write.mode(mode).format("parquet")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)
    log.info("wrote %s (partitioned by %s)", path, ", ".join(partition_by) or "-")
    return path


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.parquet(path)


def read_layer(
    spark: SparkSession, layer: str, table: str, *, lake: LakeConfig | None = None
) -> DataFrame:
    lake = lake or LAKE
    return read_parquet(spark, lake.layer(layer, table))


def write_layer(
    df: DataFrame,
    layer: str,
    table: str,
    *,
    lake: LakeConfig | None = None,
    partition_by: Sequence[str] = (),
    mode: str = "overwrite",
    repartition: int | None = None,
) -> str:
    lake = lake or LAKE
    return write_parquet(
        with_lineage(df, f"{layer}.{table}"),
        lake.layer(layer, table),
        partition_by=partition_by,
        mode=mode,
        repartition=repartition,
    )


def write_jdbc(
    df: DataFrame,
    table: str,
    *,
    warehouse: WarehouseConfig | None = None,
    mode: str = "overwrite",
    batch_size: int = 10_000,
) -> None:
    """Publish a gold table into the Postgres warehouse for dbt to build on."""
    warehouse = warehouse or WAREHOUSE
    target = f"{warehouse.schema}.{table}"
    (
        df.write.mode(mode)
        .option("batchsize", batch_size)
        .option("truncate", "true" if mode == "overwrite" else "false")
        .jdbc(warehouse.jdbc_url, target, properties=warehouse.jdbc_properties)
    )
    log.info("published %s rows to %s", df.count(), target)


def drop_lineage(df: DataFrame, keep: Iterable[str] = ()) -> DataFrame:
    keep_set = set(keep)
    return df.drop(*[c for c in df.columns if c.startswith("_") and c not in keep_set])
