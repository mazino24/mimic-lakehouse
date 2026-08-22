"""The data-quality engine itself: a gate that never fails is not a gate."""

from __future__ import annotations

import pytest

from lakehouse import dq


@pytest.fixture
def frame(spark):
    return spark.createDataFrame(
        [(1, 10, 1), (2, 20, 1), (3, None, 0), (3, 40, 5)],
        "id int, value int, label int",
    )


def test_error_severity_failure_raises_and_stops_the_pipeline(frame):
    suite = dq.Suite("silver", "demo").expect_not_null("value", severity="error")
    with pytest.raises(dq.DataQualityError) as excinfo:
        suite.run(frame, persist=False)
    assert "not_null__value" in str(excinfo.value)


def test_warn_severity_failure_is_recorded_but_lets_the_run_continue(frame):
    results = (
        dq.Suite("silver", "demo")
        .expect_not_null("value", severity="warn")
        .run(frame, persist=False)
    )
    assert results[0].passed is False
    assert results[0].severity == "warn"


def test_uniqueness_check_counts_duplicates(frame):
    result = (
        dq.Suite("silver", "demo")
        .expect_unique(["id"], severity="warn")
        .run(frame, persist=False)[0]
    )
    assert result.passed is False
    assert result.observed == 1.0


def test_accepted_values_and_ranges(frame):
    results = (
        dq.Suite("silver", "demo")
        .expect_values_in("label", [0, 1], severity="warn")
        .expect_between("value", 0, 30, severity="warn")
        .run(frame, persist=False)
    )
    assert results[0].passed is False, "label=5 is outside the accepted set"
    assert results[1].passed is False, "value=40 is outside the range"


def test_null_rate_threshold(frame):
    lenient, strict = (
        dq.Suite("silver", "demo")
        .expect_null_rate_below("value", 0.5, severity="warn")
        .expect_null_rate_below("value", 0.1, severity="warn")
        .run(frame, persist=False)
    )
    assert lenient.passed and not strict.passed


def test_class_balance_catches_a_collapsed_label(spark):
    skewed = spark.createDataFrame([(1,)] * 1 + [(0,)] * 99, "label int")
    result = (
        dq.Suite("gold", "cohort")
        .expect_class_balance("label", 0.2, 0.8, severity="warn")
        .run(skewed, persist=False)[0]
    )
    assert result.passed is False
    assert result.observed == pytest.approx(0.01)


def test_passing_suite_returns_all_green(frame):
    results = (
        dq.Suite("gold", "demo")
        .expect_row_count_between(1)
        .expect_not_null("id")
        .run(frame, persist=False)
    )
    assert all(r.passed for r in results)


def test_results_are_persisted_as_a_queryable_time_series(spark, frame, tmp_path):
    from lakehouse.config import LakeConfig

    lake = LakeConfig(root=str(tmp_path / "lake"))
    dq.Suite("gold", "demo").expect_row_count_between(1).run(frame, lake=lake)

    persisted = spark.read.parquet(lake.quality("dq_results")).collect()
    assert len(persisted) == 1
    assert persisted[0]["passed"] == "true"
    assert persisted[0]["row_count"] == 4
