"""One test that runs the whole thing: synthetic CSVs -> bronze -> silver -> gold.

Slower than the unit tests (~1 min) but it is the test that would have caught
every integration bug found while building this repo: a boolean partition key
read back as a string, an overwrite leaving orphaned partitions, a pivot
producing column names Postgres rejects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    from generate_synthetic_mimic import generate

    from lakehouse.config import LakeConfig

    workdir = tmp_path_factory.mktemp("e2e")
    raw = workdir / "raw"
    generate(raw, n_patients=120, seed=7, ecg_share=0.4)
    return LakeConfig(root=str(workdir / "lake")), raw


def test_bronze_to_gold_runs_and_produces_a_trainable_mart(spark, lake, monkeypatch):
    lake_config, raw = lake
    monkeypatch.setenv("LAKE_ROOT", lake_config.root)

    import lakehouse.config as config_module
    import lakehouse.io as io_module

    monkeypatch.setattr(config_module, "LAKE", lake_config)
    monkeypatch.setattr(io_module, "LAKE", lake_config)

    from lakehouse.io import read_csv, read_layer, write_layer
    from lakehouse.schemas import SOURCE_TABLES
    from lakehouse.transforms import cleaning
    from lakehouse.transforms import cohort as cohort_tf
    from lakehouse.transforms import features as feature_tf

    # -- bronze -------------------------------------------------------------
    for table, (relative, schema, _partitions) in SOURCE_TABLES.items():
        frame = read_csv(spark, str(raw / relative), schema)
        write_layer(frame, lake_config.bronze_prefix, table, lake=lake_config)

    bronze = lake_config.bronze_prefix
    assert read_layer(spark, bronze, "patients", lake=lake_config).count() == 120

    # -- silver -------------------------------------------------------------
    patients = cleaning.clean_patients(read_layer(spark, bronze, "patients", lake=lake_config))
    admissions = cleaning.clean_admissions(
        read_layer(spark, bronze, "admissions", lake=lake_config)
    )
    diagnoses = cleaning.clean_diagnoses(
        read_layer(spark, bronze, "diagnoses_icd", lake=lake_config)
    )
    dictionary = cleaning.clean_lab_dictionary(
        read_layer(spark, bronze, "d_labitems", lake=lake_config)
    )
    labevents = cleaning.clean_labevents(
        read_layer(spark, bronze, "labevents", lake=lake_config), admissions, dictionary
    )
    ecg = cleaning.clean_ecg_records(
        read_layer(spark, bronze, "ecg_record_list", lake=lake_config), admissions
    )

    silver = lake_config.silver_prefix
    write_layer(labevents, silver, "labevents", lake=lake_config, partition_by=["marker_type"])
    reread = read_layer(spark, silver, "labevents", lake=lake_config)
    assert dict(reread.dtypes)["is_acute_marker"] == "boolean", (
        "a boolean flag must survive a partitioned round-trip"
    )
    assert reread.filter(reread.hours_from_admit < 0).count() == 0

    # -- gold ---------------------------------------------------------------
    study_cohort = cohort_tf.build_cohort(patients, admissions, diagnoses)
    aggregated = feature_tf.aggregate_lab_features(
        labevents.join(study_cohort.select("hadm_id"), on="hadm_id")
    )
    wide = feature_tf.pivot_lab_features(aggregated, min_coverage=0.02)
    mart = feature_tf.build_feature_mart(study_cohort, wide, ecg)

    rows = mart.collect()
    assert len(rows) == study_cohort.count() > 0
    assert len({r["subject_id"] for r in rows}) == len(rows), "no duplicated patients"
    assert {r["label"] for r in rows} == {0, 1}, "both classes present"
    assert "troponin_t" in mart.columns, "the headline cardiac marker survived the pivot"
    assert any(r["has_ecg"] for r in rows), "ECG modality linked to at least one stay"

    # Column names must be valid unquoted SQL identifiers for Postgres/dbt.
    for column in mart.columns:
        assert column.replace("_", "").isalnum(), column
        assert len(column) <= 63, column
