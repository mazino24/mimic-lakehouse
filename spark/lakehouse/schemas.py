"""Explicit schemas for the raw MIMIC-IV extracts.

Reading 40 GB of CSV with ``inferSchema=True`` costs a full extra pass over the
data and silently retypes columns between runs (``hadm_id`` becomes a double as
soon as a null shows up). Every bronze reader therefore declares its schema.
"""

from __future__ import annotations

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def _s(name: str) -> StructField:
    return StructField(name, StringType(), True)


def _i(name: str) -> StructField:
    return StructField(name, IntegerType(), True)


def _d(name: str) -> StructField:
    return StructField(name, DoubleType(), True)


def _ts(name: str) -> StructField:
    return StructField(name, TimestampType(), True)


PATIENTS = StructType([
    _i("subject_id"), _s("gender"), _i("anchor_age"), _i("anchor_year"),
    _s("anchor_year_group"), _ts("dod"),
])

ADMISSIONS = StructType([
    _i("subject_id"), _i("hadm_id"), _ts("admittime"), _ts("dischtime"),
    _ts("deathtime"), _s("admission_type"), _s("admit_provider_id"),
    _s("admission_location"), _s("discharge_location"), _s("insurance"),
    _s("language"), _s("marital_status"), _s("race"), _ts("edregtime"),
    _ts("edouttime"), _i("hospital_expire_flag"),
])

DIAGNOSES_ICD = StructType([
    _i("subject_id"), _i("hadm_id"), _i("seq_num"), _s("icd_code"), _i("icd_version"),
])

D_LABITEMS = StructType([
    _i("itemid"), _s("label"), _s("fluid"), _s("category"),
])

LABEVENTS = StructType([
    StructField("labevent_id", IntegerType(), True), _i("subject_id"), _i("hadm_id"),
    _i("specimen_id"), _i("itemid"), _s("order_provider_id"), _ts("charttime"),
    _ts("storetime"), _s("value"), _d("valuenum"), _s("valueuom"),
    _d("ref_range_lower"), _d("ref_range_upper"), _s("flag"), _s("priority"),
    _s("comments"),
])

CHARTEVENTS = StructType([
    _i("subject_id"), _i("hadm_id"), _i("stay_id"), _i("caregiver_id"),
    _ts("charttime"), _ts("storetime"), _i("itemid"), _s("value"), _d("valuenum"),
    _s("valueuom"), _i("warning"),
])

ECG_RECORD_LIST = StructType([
    _i("subject_id"), _i("study_id"), _s("file_name"), _ts("ecg_time"), _s("path"),
])


#: bronze table name -> (relative source path, schema, partition columns)
SOURCE_TABLES: dict[str, tuple[str, StructType, tuple[str, ...]]] = {
    "patients": ("patients.csv", PATIENTS, ()),
    "admissions": ("admissions.csv", ADMISSIONS, ()),
    "diagnoses_icd": ("diagnoses_icd.csv", DIAGNOSES_ICD, ()),
    "d_labitems": ("d_labitems.csv", D_LABITEMS, ()),
    "labevents": ("labevents.csv", LABEVENTS, ()),
    "chartevents": ("chartevents.csv", CHARTEVENTS, ()),
    "ecg_record_list": ("mimic-iv-ecg/record_list.csv", ECG_RECORD_LIST, ()),
}
