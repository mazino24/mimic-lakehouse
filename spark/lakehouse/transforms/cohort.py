"""Silver -> gold: build the labelled study cohort.

This is where the epidemiology lives. Three rules protect the label:

* **cases**  – adult stays carrying an ICD-10 angina code (I20*, I251*);
* **controls** – adults whose *entire* record contains no circulatory (I*)
  diagnosis. Heart-failure and MI patients look like angina patients in the
  lab panel, so admitting them as controls teaches the model "cardiac vs
  non-cardiac" instead of the question actually being asked;
* **one stay per patient** – the same patient must never appear twice, or the
  train/test split leaks them across both sides.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from lakehouse.config import CLINICAL, ClinicalConfig

SPLIT_WEIGHTS = {"train": 70, "validation": 15, "test": 15}


def assign_split(df: DataFrame, key: str = "subject_id", salt: str = "angina-v1") -> DataFrame:
    """Deterministically assign train/validation/test **by patient**.

    A hash keeps the assignment stable across reruns and across machines, so a
    model trained today is comparable with one trained next month, and every
    stay belonging to a patient lands in the same split.
    """
    bucket = F.pmod(F.hash(F.concat_ws("::", F.lit(salt), F.col(key).cast("string"))), F.lit(100))
    train_max = SPLIT_WEIGHTS["train"]
    val_max = train_max + SPLIT_WEIGHTS["validation"]
    return df.withColumn("split_bucket", bucket).withColumn(
        "split",
        F.when(F.col("split_bucket") < train_max, F.lit("train"))
        .when(F.col("split_bucket") < val_max, F.lit("validation"))
        .otherwise(F.lit("test")),
    )


def _first_admission_per_patient(admissions: DataFrame) -> DataFrame:
    window = Window.partitionBy("subject_id").orderBy(
        F.col("admittime").asc(), F.col("hadm_id").asc()
    )
    return (
        admissions.withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def build_cohort(
    patients: DataFrame,
    admissions: DataFrame,
    diagnoses: DataFrame,
    clinical: ClinicalConfig | None = None,
    *,
    balance: bool = True,
) -> DataFrame:
    """Return the labelled cohort: one row per patient, ``label`` in {0, 1}."""
    clinical = clinical or CLINICAL

    adults = patients.filter(F.col("is_adult"))

    angina_stays = (
        diagnoses.filter(F.col("is_angina")).select("hadm_id").distinct()
    )
    # Patient-level exclusion, not stay-level: a control must be free of
    # circulatory disease across their whole record.
    circulatory_patients = (
        diagnoses.filter(F.col("is_circulatory")).select("subject_id").distinct()
    )

    adult_admissions = admissions.join(adults.select("subject_id"), on="subject_id", how="inner")

    cases = (
        adult_admissions.join(angina_stays, on="hadm_id", how="inner")
        .transform(_first_admission_per_patient)
        .withColumn("label", F.lit(1))
    )

    controls = (
        adult_admissions.join(circulatory_patients, on="subject_id", how="left_anti")
        .transform(_first_admission_per_patient)
        .withColumn("label", F.lit(0))
    )

    if balance:
        n_cases = cases.count()
        keep = max(int(n_cases * clinical.controls_per_case), 1)
        # Deterministic sample: order by a hash, not by rand(), so the cohort is
        # byte-identical on every rerun.
        ordering = Window.orderBy(F.hash(F.concat_ws("::", F.lit("ctl"), F.col("subject_id"))))
        controls = (
            controls.withColumn("_rank", F.row_number().over(ordering))
            .filter(F.col("_rank") <= keep)
            .drop("_rank")
        )

    cohort = cases.unionByName(controls)

    return (
        cohort.join(
            adults.select("subject_id", "gender", "anchor_age"), on="subject_id", how="inner"
        )
        .transform(assign_split)
        .select(
            "subject_id", "hadm_id", "label", "gender", "anchor_age", "admittime",
            "dischtime", "los_hours", "admit_year", "admission_type", "insurance",
            "race", "died_in_hospital", "split", "split_bucket",
        )
    )
