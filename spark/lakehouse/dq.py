"""A small data-quality engine.

Every job declares expectations against the DataFrame it just produced. Results
are persisted to ``_quality/dq_results`` in the lake (and loaded into Postgres,
where dbt exposes them as ``analytics.dq_results``), so quality is a tracked
time series rather than a log line nobody reads.

Severities
----------
``error``  -> raises ``DataQualityError`` and fails the Airflow task
``warn``   -> recorded, pipeline continues
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from lakehouse.config import LAKE, LakeConfig
from lakehouse.io import run_id

log = logging.getLogger("lakehouse.dq")

RESULT_SCHEMA = StructType([
    StructField("run_id", StringType(), False),
    StructField("checked_at", TimestampType(), False),
    StructField("layer", StringType(), False),
    StructField("table_name", StringType(), False),
    StructField("check_name", StringType(), False),
    StructField("severity", StringType(), False),
    StructField("passed", StringType(), False),
    StructField("observed", DoubleType(), True),
    StructField("threshold", DoubleType(), True),
    StructField("row_count", LongType(), True),
    StructField("details", StringType(), True),
])


class DataQualityError(RuntimeError):
    """Raised when an ``error``-severity expectation fails."""


@dataclass(frozen=True)
class CheckResult:
    check_name: str
    severity: str
    passed: bool
    observed: float | None
    threshold: float | None
    details: str = ""


@dataclass
class Expectation:
    name: str
    severity: str
    fn: Callable[[DataFrame], CheckResult]


class Suite:
    """Fluent collection of expectations for one table."""

    def __init__(self, layer: str, table: str) -> None:
        self.layer = layer
        self.table = table
        self._expectations: list[Expectation] = []

    # -- expectation builders ------------------------------------------------
    def expect_row_count_between(
        self, minimum: int, maximum: int | None = None, severity: str = "error"
    ) -> Suite:
        def check(df: DataFrame) -> CheckResult:
            count = df.count()
            ok = count >= minimum and (maximum is None or count <= maximum)
            return CheckResult(
                f"row_count_between_{minimum}_{maximum or 'inf'}",
                severity, ok, float(count), float(minimum),
                f"observed {count} rows",
            )

        return self._add(check, severity)

    def expect_not_null(self, column: str, severity: str = "error") -> Suite:
        def check(df: DataFrame) -> CheckResult:
            nulls = df.filter(F.col(column).isNull()).count()
            return CheckResult(
                f"not_null__{column}", severity, nulls == 0, float(nulls), 0.0,
                f"{nulls} null values in {column}",
            )

        return self._add(check, severity)

    def expect_unique(self, columns: Sequence[str], severity: str = "error") -> Suite:
        cols = list(columns)

        def check(df: DataFrame) -> CheckResult:
            total = df.count()
            distinct = df.select(*cols).distinct().count()
            dupes = total - distinct
            return CheckResult(
                f"unique__{'_'.join(cols)}", severity, dupes == 0, float(dupes), 0.0,
                f"{dupes} duplicate keys on {cols}",
            )

        return self._add(check, severity)

    def expect_values_in(
        self, column: str, allowed: Sequence, severity: str = "error"
    ) -> Suite:
        allowed_list = list(allowed)

        def check(df: DataFrame) -> CheckResult:
            bad = df.filter(F.col(column).isNotNull() & ~F.col(column).isin(allowed_list)).count()
            return CheckResult(
                f"values_in__{column}", severity, bad == 0, float(bad), 0.0,
                f"{bad} rows outside {allowed_list}",
            )

        return self._add(check, severity)

    def expect_between(
        self, column: str, low: float, high: float, severity: str = "error"
    ) -> Suite:
        def check(df: DataFrame) -> CheckResult:
            bad = df.filter(
                F.col(column).isNotNull() & (~F.col(column).between(low, high))
            ).count()
            return CheckResult(
                f"between__{column}", severity, bad == 0, float(bad), 0.0,
                f"{bad} rows outside [{low}, {high}]",
            )

        return self._add(check, severity)

    def expect_null_rate_below(
        self, column: str, max_rate: float, severity: str = "warn"
    ) -> Suite:
        def check(df: DataFrame) -> CheckResult:
            total = df.count()
            if total == 0:
                return CheckResult(f"null_rate__{column}", severity, False, 1.0, max_rate,
                                   "empty table")
            rate = df.filter(F.col(column).isNull()).count() / total
            return CheckResult(
                f"null_rate__{column}", severity, rate <= max_rate, rate, max_rate,
                f"null rate {rate:.3f} (max {max_rate})",
            )

        return self._add(check, severity)

    def expect_class_balance(
        self, column: str, min_share: float, max_share: float, severity: str = "warn"
    ) -> Suite:
        """Guard against a silently collapsing label distribution."""

        def check(df: DataFrame) -> CheckResult:
            total = df.count()
            if total == 0:
                return CheckResult(f"class_balance__{column}", severity, False, 0.0,
                                   min_share, "empty table")
            positives = df.filter(F.col(column) == 1).count()
            share = positives / total
            return CheckResult(
                f"class_balance__{column}", severity, min_share <= share <= max_share,
                share, min_share, f"positive share {share:.3f}",
            )

        return self._add(check, severity)

    def expect_custom(
        self, name: str, predicate: Callable[[DataFrame], CheckResult], severity: str = "error"
    ) -> Suite:
        def check(df: DataFrame) -> CheckResult:
            return predicate(df)

        return self._add(check, severity, name=name)

    # -- execution -----------------------------------------------------------
    def _add(self, fn, severity: str, name: str | None = None) -> Suite:
        self._expectations.append(Expectation(name or fn.__name__, severity, fn))
        return self

    def run(
        self,
        df: DataFrame,
        *,
        persist: bool = True,
        lake: LakeConfig | None = None,
        raise_on_error: bool = True,
    ) -> list[CheckResult]:
        df.cache()
        row_count = df.count()
        results = [exp.fn(df) for exp in self._expectations]

        for result in results:
            level = log.info if result.passed else (
                log.warning if result.severity == "warn" else log.error
            )
            level(
                "[dq] %s.%s %s -> %s (%s)",
                self.layer, self.table, result.check_name,
                "PASS" if result.passed else result.severity.upper(), result.details,
            )

        if persist:
            self._persist(df.sparkSession, results, row_count, lake or LAKE)

        failures = [r for r in results if not r.passed and r.severity == "error"]
        if failures and raise_on_error:
            raise DataQualityError(
                f"{len(failures)} blocking data-quality failure(s) on "
                f"{self.layer}.{self.table}: "
                + "; ".join(f"{r.check_name} ({r.details})" for r in failures)
            )
        return results

    def _persist(
        self, spark: SparkSession, results: list[CheckResult], row_count: int, lake: LakeConfig
    ) -> None:
        now = datetime.now(timezone.utc)
        rows = [
            (
                run_id(), now, self.layer, self.table, r.check_name, r.severity,
                "true" if r.passed else "false", r.observed, r.threshold,
                int(row_count), r.details,
            )
            for r in results
        ]
        frame = spark.createDataFrame(rows, RESULT_SCHEMA)
        (
            frame.write.mode("append")
            .partitionBy("layer", "table_name")
            .parquet(lake.quality("dq_results"))
        )
