"""Cohort construction: the rules that decide what the model is asked to learn."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from lakehouse.transforms import cohort as cohort_tf

BASE = datetime(2150, 1, 1, 10, 0, 0)


@pytest.fixture
def population(spark):
    """12 patients: 4 angina cases, 4 clean controls, 4 with other cardiac disease."""
    patients, admissions, diagnoses = [], [], []
    for i in range(1, 13):
        is_case = i <= 4
        other_cardiac = 5 <= i <= 8
        patients.append((i, "M" if i % 2 else "F", 40 + i, True, 2150, "2014 - 2016", None))
        for stay in range(2):  # every patient has two admissions
            hadm = i * 100 + stay
            admit = BASE + timedelta(days=stay * 30)
            admissions.append((
                i, hadm, admit, admit + timedelta(days=2), 48.0, 2150,
                "EW EMER.", "ER", "HOME", "Other", "SINGLE", "WHITE", False,
            ))
            code, angina, circ = ("E119", False, False)
            if is_case:
                code, angina, circ = ("I25110", True, True)
            elif other_cardiac:
                code, angina, circ = ("I5021", False, True)
            diagnoses.append((i, hadm, 1, code, 10, angina, circ, True))

    return (
        spark.createDataFrame(
            patients,
            "subject_id int, gender string, anchor_age int, is_adult boolean, "
            "anchor_year int, anchor_year_group string, dod timestamp",
        ),
        spark.createDataFrame(
            admissions,
            "subject_id int, hadm_id int, admittime timestamp, dischtime timestamp, "
            "los_hours double, admit_year int, admission_type string, "
            "admission_location string, discharge_location string, insurance string, "
            "marital_status string, race string, died_in_hospital boolean",
        ),
        spark.createDataFrame(
            diagnoses,
            "subject_id int, hadm_id int, seq_num int, icd_code string, icd_version int, "
            "is_angina boolean, is_circulatory boolean, is_primary boolean",
        ),
    )


def test_controls_exclude_every_patient_with_cardiac_disease(population):
    result = cohort_tf.build_cohort(*population, balance=False).collect()
    controls = {r["subject_id"] for r in result if r["label"] == 0}

    # Patients 5-8 have heart failure: they are neither cases nor controls.
    assert controls == {9, 10, 11, 12}
    assert not controls & {5, 6, 7, 8}


def test_one_row_per_patient_and_it_is_their_first_admission(population):
    result = cohort_tf.build_cohort(*population, balance=False).collect()
    subjects = [r["subject_id"] for r in result]

    assert len(subjects) == len(set(subjects)), "a patient must never appear twice"
    for row in result:
        assert row["hadm_id"] % 100 == 0, "the earlier of the two stays is kept"


def test_cases_and_controls_are_labelled_correctly(population):
    result = cohort_tf.build_cohort(*population, balance=False).collect()
    labels = {r["subject_id"]: r["label"] for r in result}

    assert all(labels[i] == 1 for i in (1, 2, 3, 4))
    assert all(labels[i] == 0 for i in (9, 10, 11, 12))


def test_balancing_matches_control_count_to_case_count(population):
    result = cohort_tf.build_cohort(*population, balance=True).collect()
    cases = sum(1 for r in result if r["label"] == 1)
    controls = sum(1 for r in result if r["label"] == 0)

    assert cases == controls == 4


def test_split_is_deterministic_across_runs(population):
    first = {r["subject_id"]: r["split"] for r in cohort_tf.build_cohort(*population).collect()}
    second = {r["subject_id"]: r["split"] for r in cohort_tf.build_cohort(*population).collect()}

    assert first == second, "reruns must reproduce the same split, byte for byte"
    assert set(first.values()) <= {"train", "validation", "test"}


def test_split_assignment_is_by_patient_not_by_row(spark):
    df = spark.createDataFrame(
        [(7, 1), (7, 2), (7, 3), (8, 4)], "subject_id int, hadm_id int"
    )
    splits = cohort_tf.assign_split(df).collect()
    by_subject = {}
    for row in splits:
        by_subject.setdefault(row["subject_id"], set()).add(row["split"])

    assert all(len(v) == 1 for v in by_subject.values()), "no patient straddles two splits"


def test_minors_are_excluded(spark, population):
    patients, admissions, diagnoses = population
    minors = patients.withColumn("is_adult", patients.anchor_age > 200)  # nobody qualifies
    assert cohort_tf.build_cohort(minors, admissions, diagnoses, balance=False).count() == 0
