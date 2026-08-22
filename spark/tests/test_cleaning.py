"""Silver-layer rules. Each test pins a bug that was real in the source data."""

from __future__ import annotations

from datetime import datetime

from lakehouse.transforms import cleaning

ADMIT_SCHEMA = (
    "subject_id int, hadm_id int, admittime timestamp, dischtime timestamp, "
    "deathtime timestamp, admission_type string, admit_provider_id string, "
    "admission_location string, discharge_location string, insurance string, "
    "language string, marital_status string, race string, edregtime timestamp, "
    "edouttime timestamp, hospital_expire_flag int"
)


def dt(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def _admission(subject_id, hadm_id, admit, discharge):
    return (
        subject_id, hadm_id, dt(admit), dt(discharge), None, "EW EMER.", "P1", "ER",
        "HOME", "Other", "ENGLISH", "SINGLE", "WHITE", None, None, 0,
    )


def test_patients_deduplicated_and_gender_normalised(spark):
    df = spark.createDataFrame(
        [
            (1, "f", 64, 2150, "2014 - 2016", None),
            (1, "F", 64, 2151, "2014 - 2016", None),  # duplicate subject
            (2, "MALE", 17, 2160, "2014 - 2016", None),
            (3, "M", 205, 2160, "2014 - 2016", None),  # impossible age
        ],
        "subject_id int, gender string, anchor_age int, anchor_year int, "
        "anchor_year_group string, dod timestamp",
    )
    out = {r["subject_id"]: r for r in cleaning.clean_patients(df).collect()}

    assert len(out) == 3, "one row per subject_id"
    assert out[1]["gender"] == "F"
    assert out[2]["gender"] == "M"
    assert out[2]["is_adult"] is False, "17-year-olds are not adults"
    assert out[3]["anchor_age"] is None, "out-of-range age is nulled, not kept"


def test_admissions_drop_impossible_time_windows(spark):
    df = spark.createDataFrame(
        [
            _admission(1, 100, "2150-01-01 10:00:00", "2150-01-03 10:00:00"),
            _admission(2, 200, "2150-01-05 10:00:00", "2150-01-04 10:00:00"),  # corrupt
            _admission(1, 100, "2150-01-01 10:00:00", "2150-01-03 10:00:00"),  # duplicate
        ],
        ADMIT_SCHEMA,
    )
    out = cleaning.clean_admissions(df).collect()

    assert [r["hadm_id"] for r in out] == [100], "corrupt and duplicate stays removed"
    assert out[0]["los_hours"] == 48.0


def test_icd_codes_normalised_and_angina_flagged(spark):
    df = spark.createDataFrame(
        [
            (1, 100, 1, "I25.110", 10),   # dotted form
            (2, 200, 1, " I2510 ", 10),   # whitespace
            (3, 300, 1, "I209", 10),      # angina pectoris
            (4, 400, 1, "I5021", 10),     # heart failure: circulatory, not angina
            (5, 500, 1, "E119", 10),      # diabetes: neither
            (6, 600, 1, "4139", 9),       # ICD-9 angina: out of scope
        ],
        "subject_id int, hadm_id int, seq_num int, icd_code string, icd_version int",
    )
    out = {r["subject_id"]: r for r in cleaning.clean_diagnoses(df).collect()}

    assert out[1]["icd_code"] == "I25110"
    assert out[2]["icd_code"] == "I2510"
    # The original pipeline used the prefix "I2511", which missed I25.10 —
    # exactly the code this asserts on.
    assert out[1]["is_angina"] and out[2]["is_angina"] and out[3]["is_angina"]
    assert not out[4]["is_angina"] and out[4]["is_circulatory"]
    assert not out[5]["is_circulatory"]
    assert not out[6]["is_angina"], "ICD-9 rows must not be flagged"


def test_labs_outside_the_admission_window_are_dropped(spark):
    labs = spark.createDataFrame(
        [
            (1, 100, 51003, dt("2150-01-01 12:00:00"), 0.31, "ng/mL", ""),   # in window
            (2, 100, 51003, dt("2149-06-01 12:00:00"), 0.02, "ng/mL", ""),   # months earlier
            (3, 100, 51003, dt("2150-02-01 12:00:00"), 0.05, "ng/mL", ""),   # after discharge
            (4, 100, 99999, dt("2150-01-01 13:00:00"), 7.0, "mg/dL", ""),    # not a target lab
            (5, 100, 51003, dt("2150-01-01 12:00:00"), None, "ng/mL", ""),   # text-only result
        ],
        "labevent_id int, hadm_id int, itemid int, charttime timestamp, valuenum double, "
        "valueuom string, flag string",
    )
    admissions = spark.createDataFrame(
        [(1, 100, dt("2150-01-01 10:00:00"), dt("2150-01-03 10:00:00"))],
        "subject_id int, hadm_id int, admittime timestamp, dischtime timestamp",
    )
    dictionary = spark.createDataFrame(
        [
            (51003, "Troponin T", "Blood", "Chemistry", True, True),
            (99999, "Bilirubin", "Blood", "Chemistry", False, False),
        ],
        "itemid int, label string, fluid string, category string, "
        "is_target_lab boolean, is_acute_marker boolean",
    )
    out = cleaning.clean_labevents(labs, admissions, dictionary).collect()

    assert len(out) == 1
    assert out[0]["valuenum"] == 0.31
    assert out[0]["hours_from_admit"] == 2.0
    assert out[0]["marker_type"] == "acute"


def test_ecg_matched_to_the_stay_that_contains_it(spark):
    records = spark.createDataFrame(
        [
            (1, 900, "900", dt("2150-01-01 11:00:00"), "files/p1/s900/900"),
            (1, 901, "901", dt("2150-01-02 11:00:00"), "files/p1/s901/901"),
            (1, 902, "902", dt("2151-01-02 11:00:00"), "files/p1/s902/902"),  # other year
        ],
        "subject_id int, study_id int, file_name string, ecg_time timestamp, path string",
    )
    admissions = spark.createDataFrame(
        [(1, 100, dt("2150-01-01 10:00:00"), dt("2150-01-03 10:00:00"))],
        "subject_id int, hadm_id int, admittime timestamp, dischtime timestamp",
    )
    out = cleaning.clean_ecg_records(records, admissions).collect()

    assert len(out) == 1, "one ECG per stay"
    assert out[0]["study_id"] == 900, "the earliest in-window study wins"
