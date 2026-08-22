#!/usr/bin/env python3
"""Job 3 — silver -> gold.

Produces the three tables the rest of the organisation consumes:

``gold.cohort``            labelled study population, one row per patient
``gold.lab_features``      per-admission aggregated labs (long form)
``gold.feature_mart``      the modelling table: cohort x features, one row/stay
``gold.feature_coverage``  per-column null rates, for monitoring drift

    spark-submit jobs/gold_build.py --min-coverage 0.02
"""

from __future__ import annotations

import argparse
import sys

from lakehouse import dq
from lakehouse.config import CLINICAL, LAKE
from lakehouse.io import read_layer, write_layer
from lakehouse.session import build_session, configure_logging
from lakehouse.transforms import cohort as cohort_tf
from lakehouse.transforms import features as feature_tf

log = configure_logging()


def build_gold(spark, *, min_coverage: float, balance: bool = True) -> dict[str, int]:
    silver, gold = LAKE.silver_prefix, LAKE.gold_prefix
    counts: dict[str, int] = {}

    patients = read_layer(spark, silver, "patients")
    admissions = read_layer(spark, silver, "admissions")
    diagnoses = read_layer(spark, silver, "diagnoses")
    labevents = read_layer(spark, silver, "labevents")

    try:
        ecg = read_layer(spark, silver, "ecg_records")
    except Exception:
        log.warning("no silver.ecg_records found; building tabular-only mart")
        ecg = None

    study_cohort = cohort_tf.build_cohort(
        patients, admissions, diagnoses, CLINICAL, balance=balance
    )
    (
        dq.Suite(gold, "cohort")
        .expect_row_count_between(2)
        .expect_unique(["subject_id"])
        .expect_unique(["hadm_id"])
        .expect_not_null("label")
        .expect_values_in("label", [0, 1])
        .expect_values_in("split", ["train", "validation", "test"])
        .expect_between("anchor_age", CLINICAL.min_age, 120)
        .expect_class_balance("label", 0.2, 0.8)
        # The whole point of splitting by patient: no subject in two splits.
        .expect_custom(
            "no_patient_across_splits",
            lambda df: dq.CheckResult(
                "no_patient_across_splits", "error",
                df.select("subject_id").distinct().count() == df.count(),
                float(df.count() - df.select("subject_id").distinct().count()), 0.0,
                "patients appearing in more than one split",
            ),
        )
        .run(study_cohort)
    )
    write_layer(study_cohort, gold, "cohort", partition_by=["split"])
    counts["cohort"] = study_cohort.count()

    cohort_stays = study_cohort.select("hadm_id")
    cohort_labs = labevents.join(cohort_stays, on="hadm_id", how="inner")

    lab_features = feature_tf.aggregate_lab_features(cohort_labs)
    (
        dq.Suite(gold, "lab_features")
        .expect_not_null("hadm_id")
        .expect_unique(["hadm_id", "itemid"])
        .expect_values_in("aggregation", ["first", "mean"])
        .run(lab_features)
    )
    write_layer(lab_features, gold, "lab_features")
    counts["lab_features"] = lab_features.count()

    wide = feature_tf.pivot_lab_features(lab_features, min_coverage=min_coverage)
    mart = feature_tf.build_feature_mart(study_cohort, wide, ecg)
    (
        dq.Suite(gold, "feature_mart")
        .expect_unique(["hadm_id"])
        .expect_not_null("label")
        .expect_class_balance("label", 0.2, 0.8)
        .expect_null_rate_below("age", 0.0, severity="error")
        .run(mart)
    )
    write_layer(mart, gold, "feature_mart", partition_by=["split"])
    counts["feature_mart"] = mart.count()

    lab_columns = tuple(c for c in wide.columns if c != "hadm_id")
    coverage = feature_tf.feature_coverage_report(mart, lab_columns=lab_columns)
    write_layer(coverage, gold, "feature_coverage")
    counts["feature_coverage"] = coverage.count()

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the gold layer")
    parser.add_argument(
        "--min-coverage", type=float, default=0.02,
        help="Drop lab columns measured in fewer than this share of admissions",
    )
    parser.add_argument("--no-balance", action="store_true",
                        help="Keep every eligible control instead of matching case count")
    args = parser.parse_args(argv)

    spark = build_session("mimic-gold-build")
    counts = build_gold(spark, min_coverage=args.min_coverage, balance=not args.no_balance)
    spark.stop()
    log.info("gold complete: %s", ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
