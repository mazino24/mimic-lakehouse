"""Bronze -> silver: conform, type, deduplicate, and flag clinical concepts.

Silver keeps every admission (no cohort filtering yet) so the layer stays
reusable for other studies built on the same lake.
"""

from __future__ import annotations

from collections.abc import Sequence

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from lakehouse.config import CLINICAL, ClinicalConfig


def _icd_prefix_filter(column, prefixes: Sequence[str]):
    """``startswith`` over a tuple of prefixes, as a single boolean column."""
    condition = F.lit(False)
    for prefix in prefixes:
        condition = condition | column.startswith(prefix)
    return condition


def clean_patients(patients: DataFrame) -> DataFrame:
    """One row per ``subject_id`` with a normalised gender flag."""
    window = Window.partitionBy("subject_id").orderBy(F.col("anchor_year").asc_nulls_last())
    return (
        patients.filter(F.col("subject_id").isNotNull())
        .withColumn("gender", F.upper(F.trim(F.col("gender"))))
        .withColumn(
            "gender",
            F.when(F.col("gender").isin("M", "MALE"), F.lit("M"))
            .when(F.col("gender").isin("F", "FEMALE"), F.lit("F"))
            .otherwise(F.lit(None)),
        )
        # Ages above 89 are shifted to 91 by MIMIC's de-identification policy;
        # anything outside [0, 120] is a data error, not a very old patient.
        .withColumn(
            "anchor_age",
            F.when(F.col("anchor_age").between(0, 120), F.col("anchor_age")),
        )
        .withColumn("is_adult", (F.col("anchor_age") >= F.lit(CLINICAL.min_age)))
        .withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .select(
            "subject_id", "gender", "anchor_age", "is_adult",
            "anchor_year", "anchor_year_group", "dod",
        )
    )


def clean_admissions(admissions: DataFrame) -> DataFrame:
    """One row per ``hadm_id``; drops stays with an impossible time window."""
    window = Window.partitionBy("hadm_id").orderBy(F.col("admittime").asc_nulls_last())
    cleaned = (
        admissions.filter(F.col("hadm_id").isNotNull() & F.col("subject_id").isNotNull())
        .filter(F.col("admittime").isNotNull())
        # A discharge before admission means a corrupt record: exclude rather
        # than silently produce negative length-of-stay features.
        .filter(F.col("dischtime").isNull() | (F.col("dischtime") >= F.col("admittime")))
        .withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .withColumn(
            "los_hours",
            F.round(
                (F.col("dischtime").cast("long") - F.col("admittime").cast("long")) / 3600.0, 2
            ),
        )
        .withColumn("admit_year", F.year("admittime"))
        .withColumn("died_in_hospital", F.coalesce(F.col("hospital_expire_flag"), F.lit(0)) == 1)
    )
    return cleaned.select(
        "subject_id", "hadm_id", "admittime", "dischtime", "los_hours", "admit_year",
        "admission_type", "admission_location", "discharge_location", "insurance",
        "marital_status", "race", "died_in_hospital",
    )


def clean_diagnoses(
    diagnoses: DataFrame, clinical: ClinicalConfig | None = None
) -> DataFrame:
    """Normalise ICD codes and flag angina / circulatory diagnoses.

    MIMIC-IV stores ICD codes without dots and occasionally with trailing
    whitespace, so ``I25.110`` may arrive as ``"I25110 "``.
    """
    clinical = clinical or CLINICAL
    normalised = (
        diagnoses.filter(F.col("hadm_id").isNotNull() & F.col("icd_code").isNotNull())
        .withColumn("icd_code", F.upper(F.regexp_replace(F.trim(F.col("icd_code")), r"\.", "")))
        .withColumn("icd_version", F.coalesce(F.col("icd_version"), F.lit(10)))
    )
    icd10 = F.col("icd_version") == 10
    return (
        normalised.withColumn(
            "is_angina",
            icd10 & _icd_prefix_filter(F.col("icd_code"), clinical.angina_icd10_prefixes),
        )
        .withColumn(
            "is_circulatory",
            icd10 & F.col("icd_code").startswith(clinical.cardiac_icd10_prefix),
        )
        .withColumn("is_primary", F.coalesce(F.col("seq_num"), F.lit(99)) == 1)
        .dropDuplicates(["subject_id", "hadm_id", "icd_code", "icd_version"])
        .select(
            "subject_id", "hadm_id", "seq_num", "icd_code", "icd_version",
            "is_angina", "is_circulatory", "is_primary",
        )
    )


def clean_lab_dictionary(
    d_labitems: DataFrame, clinical: ClinicalConfig | None = None
) -> DataFrame:
    """Restrict the lab dictionary to the panels the study uses."""
    clinical = clinical or CLINICAL
    label = F.lower(F.trim(F.col("label")))
    is_target = F.lit(False)
    for pattern in clinical.target_lab_patterns:
        is_target = is_target | label.contains(pattern)
    is_acute = F.lit(False)
    for pattern in clinical.acute_marker_patterns:
        is_acute = is_acute | label.contains(pattern)

    return (
        d_labitems.filter(F.col("label").isNotNull())
        .withColumn("label", F.trim(F.col("label")))
        .withColumn("is_target_lab", is_target)
        .withColumn("is_acute_marker", is_acute)
        .dropDuplicates(["itemid"])
        .select("itemid", "label", "fluid", "category", "is_target_lab", "is_acute_marker")
    )


def clean_labevents(
    labevents: DataFrame,
    admissions: DataFrame,
    lab_dictionary: DataFrame,
    clinical: ClinicalConfig | None = None,
) -> DataFrame:
    """The heavy join: 40 GB of lab results narrowed to in-window target labs.

    Three correctness rules are enforced here, each of which was a real bug in
    the original notebook-style pipeline:

    1. only labs of interest survive (dictionary join, broadcast);
    2. only measurements drawn *between admission and discharge* count — an
       outpatient draw three months later is not a feature of this stay;
    3. ``valuenum`` must parse, otherwise the row is a text result.
    """
    clinical = clinical or CLINICAL
    target_items = lab_dictionary.filter(F.col("is_target_lab"))

    windows = admissions.select("hadm_id", "subject_id", "admittime", "dischtime")

    events = (
        labevents.filter(F.col("hadm_id").isNotNull() & F.col("valuenum").isNotNull())
        .filter(F.col("charttime").isNotNull())
        .join(F.broadcast(target_items), on="itemid", how="inner")
        .join(windows.drop("subject_id"), on="hadm_id", how="inner")
    )

    if clinical.require_labs_within_admission:
        events = events.filter(
            (F.col("charttime") >= F.col("admittime"))
            & (F.col("charttime") <= F.coalesce(F.col("dischtime"), F.col("charttime")))
        )

    return (
        events.withColumn(
            "hours_from_admit",
            F.round(
                (F.col("charttime").cast("long") - F.col("admittime").cast("long")) / 3600.0, 3
            ),
        )
        # Partition key is a string, not the boolean: Hive-style partition
        # values round-trip as text, so partitioning on a boolean column would
        # hand downstream jobs a StringType where they expect BooleanType.
        .withColumn(
            "marker_type",
            F.when(F.col("is_acute_marker"), F.lit("acute")).otherwise(F.lit("routine")),
        )
        .dropDuplicates(["hadm_id", "itemid", "charttime", "valuenum"])
        .select(
            "hadm_id", "itemid", "label", "category", "is_acute_marker", "marker_type",
            "charttime", "hours_from_admit", "valuenum", "valueuom", "flag",
        )
    )


def clean_ecg_records(
    record_list: DataFrame, admissions: DataFrame, max_hours_from_admit: int = 72
) -> DataFrame:
    """Link ECG studies to the admission they belong to.

    ``record_list.csv`` has no ``hadm_id``: the study is matched to the stay
    whose window contains ``ecg_time`` (with a grace period for ED ECGs taken
    just before the formal admission timestamp).
    """
    stays = admissions.select("subject_id", "hadm_id", "admittime", "dischtime")
    joined = (
        record_list.filter(F.col("subject_id").isNotNull() & F.col("ecg_time").isNotNull())
        .join(stays, on="subject_id", how="inner")
        .filter(
            (F.col("ecg_time") >= F.col("admittime") - F.expr("INTERVAL 12 HOURS"))
            & (F.col("ecg_time") <= F.coalesce(F.col("dischtime"), F.col("ecg_time")))
        )
        .withColumn(
            "hours_from_admit",
            F.round((F.col("ecg_time").cast("long") - F.col("admittime").cast("long")) / 3600.0, 3),
        )
        .filter(F.col("hours_from_admit") <= F.lit(max_hours_from_admit))
    )
    # One ECG per stay: the earliest study, i.e. the one taken at presentation.
    earliest = Window.partitionBy("hadm_id").orderBy(
        F.abs(F.col("hours_from_admit")).asc(), F.col("study_id").asc()
    )
    return (
        joined.withColumn("_rn", F.row_number().over(earliest))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .select("subject_id", "hadm_id", "study_id", "file_name", "path", "ecg_time",
                "hours_from_admit")
    )
