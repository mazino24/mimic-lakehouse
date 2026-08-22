#!/usr/bin/env python3
"""Job 2 — bronze -> silver.

Conformed, deduplicated, clinically annotated entities. No cohort logic yet:
silver is the layer another study on the same lake would reuse.

    spark-submit jobs/silver_transform.py
"""

from __future__ import annotations

import argparse
import sys

from lakehouse import dq
from lakehouse.config import LAKE
from lakehouse.io import read_layer, write_layer
from lakehouse.session import build_session, configure_logging
from lakehouse.transforms import cleaning

log = configure_logging()


def build_silver(spark, *, with_ecg: bool = True) -> dict[str, int]:
    bronze = LAKE.bronze_prefix
    silver = LAKE.silver_prefix
    counts: dict[str, int] = {}

    patients = cleaning.clean_patients(read_layer(spark, bronze, "patients"))
    (
        dq.Suite(silver, "patients")
        .expect_unique(["subject_id"])
        .expect_not_null("subject_id")
        .expect_values_in("gender", ["M", "F"], severity="warn")
        .expect_between("anchor_age", 0, 120)
        .run(patients)
    )
    write_layer(patients, silver, "patients")
    counts["patients"] = patients.count()

    admissions = cleaning.clean_admissions(read_layer(spark, bronze, "admissions"))
    (
        dq.Suite(silver, "admissions")
        .expect_unique(["hadm_id"])
        .expect_not_null("admittime")
        .expect_between("los_hours", 0, 24 * 365)
        .run(admissions)
    )
    write_layer(admissions, silver, "admissions", partition_by=["admit_year"])
    counts["admissions"] = admissions.count()

    diagnoses = cleaning.clean_diagnoses(read_layer(spark, bronze, "diagnoses_icd"))
    (
        dq.Suite(silver, "diagnoses")
        .expect_not_null("icd_code")
        .expect_unique(["subject_id", "hadm_id", "icd_code", "icd_version"])
        .run(diagnoses)
    )
    write_layer(diagnoses, silver, "diagnoses")
    counts["diagnoses"] = diagnoses.count()

    lab_dictionary = cleaning.clean_lab_dictionary(read_layer(spark, bronze, "d_labitems"))
    write_layer(lab_dictionary, silver, "lab_dictionary")
    counts["lab_dictionary"] = lab_dictionary.count()

    labevents = cleaning.clean_labevents(
        read_layer(spark, bronze, "labevents"), admissions, lab_dictionary
    )
    (
        dq.Suite(silver, "labevents")
        .expect_row_count_between(1)
        .expect_not_null("hadm_id")
        .expect_not_null("valuenum")
        # Every surviving measurement must sit inside its admission window.
        .expect_custom(
            "labs_within_admission_window",
            lambda df: dq.CheckResult(
                "labs_within_admission_window", "error",
                df.filter(df.hours_from_admit < 0).count() == 0,
                float(df.filter(df.hours_from_admit < 0).count()), 0.0,
                "lab measurements charted before admission",
            ),
        )
        .run(labevents)
    )
    write_layer(labevents, silver, "labevents", partition_by=["marker_type"])
    counts["labevents"] = labevents.count()

    if with_ecg:
        try:
            ecg = cleaning.clean_ecg_records(
                read_layer(spark, bronze, "ecg_record_list"), admissions
            )
            dq.Suite(silver, "ecg_records").expect_unique(["hadm_id"]).run(ecg)
            write_layer(ecg, silver, "ecg_records")
            counts["ecg_records"] = ecg.count()
        except Exception as exc:
            log.warning("ECG record list unavailable, skipping silver.ecg_records: %s", exc)

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the silver layer")
    parser.add_argument("--no-ecg", action="store_true", help="Skip the ECG modality")
    args = parser.parse_args(argv)

    spark = build_session("mimic-silver-transform")
    counts = build_silver(spark, with_ecg=not args.no_ecg)
    spark.stop()
    log.info("silver complete: %s", ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
