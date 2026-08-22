"""Feature aggregation: the first-vs-mean rule is the crux of the pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from lakehouse.transforms import features as feature_tf

BASE = datetime(2150, 1, 1, 10, 0, 0)


def _lab_rows(rows):
    """rows: (hadm_id, itemid, label, is_acute, hours, value)"""
    return [
        (h, i, label, acute, "acute" if acute else "routine",
         BASE + timedelta(hours=hours), float(hours), float(value), "mg/dL", "")
        for h, i, label, acute, hours, value in rows
    ]


LAB_SCHEMA = (
    "hadm_id int, itemid int, label string, is_acute_marker boolean, marker_type string, "
    "charttime timestamp, hours_from_admit double, valuenum double, valueuom string, flag string"
)


@pytest.fixture
def labs(spark):
    return spark.createDataFrame(
        _lab_rows([
            # Serial troponin: rises at presentation then falls after treatment.
            (100, 51003, "Troponin T", True, 1, 0.40),
            (100, 51003, "Troponin T", True, 8, 0.20),
            (100, 51003, "Troponin T", True, 20, 0.06),
            # Stable marker measured twice.
            (100, 50931, "Glucose", False, 2, 100.0),
            (100, 50931, "Glucose", False, 30, 140.0),
        ]),
        LAB_SCHEMA,
    )


def test_acute_marker_uses_the_first_draw_not_the_mean(labs):
    out = {r["itemid"]: r for r in feature_tf.aggregate_lab_features(labs).collect()}
    troponin = out[51003]

    assert troponin["aggregation"] == "first"
    assert troponin["feature_value"] == pytest.approx(0.40)
    # The mean (0.22) would understate a clearly positive troponin.
    assert troponin["mean_value"] == pytest.approx(0.22)
    assert troponin["measurement_count"] == 3
    assert troponin["first_measurement_hours"] == pytest.approx(1.0)


def test_stable_marker_uses_the_mean(labs):
    glucose = {r["itemid"]: r for r in feature_tf.aggregate_lab_features(labs).collect()}[50931]

    assert glucose["aggregation"] == "mean"
    assert glucose["feature_value"] == pytest.approx(120.0)


def test_pivot_produces_one_row_per_admission_with_safe_column_names(spark, labs):
    aggregated = feature_tf.aggregate_lab_features(labs)
    wide = feature_tf.pivot_lab_features(aggregated)

    assert wide.count() == 1
    assert set(wide.columns) == {"hadm_id", "troponin_t", "glucose"}
    row = wide.collect()[0]
    assert row["troponin_t"] == pytest.approx(0.40)


def test_coverage_filter_drops_barely_measured_labs(spark):
    rows = _lab_rows(
        [(h, 50931, "Glucose", False, 1, 100.0) for h in range(1, 11)]
        + [(1, 51100, "Creatinine, Ascites", False, 1, 1.2)]  # 1 of 10 admissions
    )
    aggregated = feature_tf.aggregate_lab_features(spark.createDataFrame(rows, LAB_SCHEMA))
    wide = feature_tf.pivot_lab_features(aggregated, min_coverage=0.5)

    assert "glucose" in wide.columns
    assert "creatinine_ascites" not in wide.columns


def test_feature_mart_keeps_nulls_and_never_imputes(spark, labs):
    cohort = spark.createDataFrame(
        [(1, 100, 1, "M", 64, "train"), (2, 200, 0, "F", 55, "test")],
        "subject_id int, hadm_id int, label int, gender string, anchor_age int, split string",
    )
    wide = feature_tf.pivot_lab_features(feature_tf.aggregate_lab_features(labs))
    mart = {r["hadm_id"]: r for r in feature_tf.build_feature_mart(cohort, wide).collect()}

    assert mart[200]["troponin_t"] is None, "a missing test stays missing"
    assert mart[100]["gender_male"] == 1 and mart[200]["gender_male"] == 0
    assert mart[100]["age"] == 64


def test_coverage_report_measures_null_rates(spark, labs):
    cohort = spark.createDataFrame(
        [(1, 100, 1, "M", 64, "train"), (2, 200, 0, "F", 55, "test")],
        "subject_id int, hadm_id int, label int, gender string, anchor_age int, split string",
    )
    wide = feature_tf.pivot_lab_features(feature_tf.aggregate_lab_features(labs))
    mart = feature_tf.build_feature_mart(cohort, wide)
    report = {
        r["column_name"]: r
        for r in feature_tf.feature_coverage_report(
            mart, lab_columns=("troponin_t", "glucose")
        ).collect()
    }

    assert report["troponin_t"]["null_rate"] == pytest.approx(0.5)
    assert report["age"]["null_rate"] == pytest.approx(0.0)
    # Labs and attributes are distinguished so the "nobody imputed upstream"
    # test can ignore demographics, which are legitimately complete.
    assert report["troponin_t"]["column_kind"] == "lab_feature"
    assert report["age"]["column_kind"] == "attribute"
