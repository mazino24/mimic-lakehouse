"""Gold: turn per-measurement lab rows into one modelling row per admission."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from lakehouse.transforms.naming import dedupe_columns, normalize_column


def aggregate_lab_features(labs: DataFrame) -> DataFrame:
    """Collapse repeated measurements to one value per (admission, lab).

    Aggregation depends on the marker's clinical behaviour:

    * **acute markers** (Troponin, CK-MB) -> the *first* in-window draw. These
      are diagnostic at presentation; averaging serial draws taken after
      treatment started dilutes exactly the signal being measured.
    * **stable markers** (HDL, creatinine, glucose, ...) -> the mean over the
      stay.

    ``measurement_count`` and ``hours_from_admit`` are kept because
    *how often* a test was ordered is itself clinical signal.
    """
    first_value_window = Window.partitionBy("hadm_id", "itemid").orderBy(
        F.col("charttime").asc(), F.col("valuenum").asc()
    )
    with_first = labs.withColumn(
        "_first_value",
        F.first(F.col("valuenum")).over(first_value_window),
    ).withColumn(
        "_first_charttime",
        F.first(F.col("charttime")).over(first_value_window),
    )

    return (
        with_first.groupBy("hadm_id", "itemid", "label", "is_acute_marker")
        .agg(
            F.first("_first_value").alias("first_value"),
            F.avg("valuenum").alias("mean_value"),
            F.max("valuenum").alias("max_value"),
            F.min("valuenum").alias("min_value"),
            F.count(F.lit(1)).cast("int").alias("measurement_count"),
            F.min("hours_from_admit").alias("first_measurement_hours"),
        )
        .withColumn(
            "feature_value",
            F.when(F.col("is_acute_marker"), F.col("first_value")).otherwise(F.col("mean_value")),
        )
        .withColumn(
            "aggregation",
            F.when(F.col("is_acute_marker"), F.lit("first")).otherwise(F.lit("mean")),
        )
    )


def pivot_lab_features(aggregated: DataFrame, *, min_coverage: float = 0.0) -> DataFrame:
    """Long -> wide: one column per lab, one row per admission.

    ``min_coverage`` drops labs measured in fewer than that share of
    admissions; a column that is 99.7 % null is noise in a warehouse table and
    an invitation to leak imputation statistics in the model.
    """
    labels = [
        row["label"]
        for row in aggregated.select("label").distinct().orderBy("label").collect()
    ]
    if not labels:
        return aggregated.select("hadm_id").distinct()

    wide = (
        aggregated.groupBy("hadm_id")
        .pivot("label", labels)
        .agg(F.first("feature_value"))
    )

    if min_coverage > 0:
        total = wide.count()
        if total:
            coverage = wide.select(
                *[
                    (F.count(F.col(f"`{label}`")) / F.lit(total)).alias(label)
                    for label in labels
                ]
            ).collect()[0].asDict()
            labels = [label for label in labels if (coverage.get(label) or 0) >= min_coverage]
            wide = wide.select("hadm_id", *[F.col(f"`{label}`") for label in labels])

    renamed = dedupe_columns([normalize_column(label) for label in labels])
    for source, target in zip(labels, renamed, strict=False):
        wide = wide.withColumnRenamed(source, target)
    return wide


def build_feature_mart(
    cohort: DataFrame,
    lab_features: DataFrame,
    ecg_records: DataFrame | None = None,
) -> DataFrame:
    """Join the cohort to its features — the table the models train on.

    Missing values are deliberately left as NULL. Absence of a test is
    clinically meaningful, tree models handle NULL natively, and imputing here
    would fit statistics over train *and* test rows at once.
    """
    mart = (
        cohort.join(lab_features, on="hadm_id", how="left")
        .withColumn("gender_male", F.when(F.col("gender") == "M", 1).otherwise(0))
        .withColumn("age", F.col("anchor_age"))
    )

    if ecg_records is not None:
        mart = mart.join(
            ecg_records.select(
                "hadm_id",
                F.col("study_id").alias("ecg_study_id"),
                F.col("path").alias("ecg_path"),
                F.col("hours_from_admit").alias("ecg_hours_from_admit"),
            ),
            on="hadm_id",
            how="left",
        ).withColumn("has_ecg", F.col("ecg_study_id").isNotNull())
    else:
        mart = mart.withColumn("has_ecg", F.lit(False))

    feature_columns = [
        c
        for c in mart.columns
        if c not in {
            "subject_id", "hadm_id", "label", "split", "split_bucket", "gender",
            "admittime", "dischtime", "ecg_path", "ecg_study_id",
        }
        and not c.startswith("_")
    ]
    return mart.withColumn("feature_count", F.lit(len(feature_columns)))


def feature_coverage_report(
    mart: DataFrame,
    lab_columns: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> DataFrame:
    """Per-column null rate — published to the warehouse for monitoring.

    Columns are classified as ``lab_feature`` or ``attribute``. The
    distinction matters downstream: a demographic with zero nulls is normal,
    whereas a *lab* with zero nulls means somebody imputed upstream, and there
    is a dbt test that fails the build when that happens.
    """
    skip = set(exclude) | {"subject_id", "hadm_id", "label", "split"}
    columns = [c for c in mart.columns if c not in skip and not c.startswith("_")]
    lab_set = set(lab_columns)
    total = mart.count()
    rows = mart.select(
        *[F.count(F.col(f"`{c}`")).alias(c) for c in columns]
    ).collect()[0].asDict()
    spark = mart.sparkSession
    return spark.createDataFrame(
        [
            (
                column,
                "lab_feature" if column in lab_set else "attribute",
                int(non_null),
                int(total),
                round(1 - (non_null / total), 6) if total else 1.0,
            )
            for column, non_null in rows.items()
        ],
        "column_name string, column_kind string, non_null_rows bigint, "
        "total_rows bigint, null_rate double",
    )
