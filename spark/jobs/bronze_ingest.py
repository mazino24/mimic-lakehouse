#!/usr/bin/env python3
"""Job 1 — raw CSV -> bronze Parquet.

Bronze is a faithful, *typed* copy of the source: no filtering, no business
logic, no renames. Its only jobs are (a) getting off CSV so every downstream
read is columnar and predicate-pushed, and (b) recording lineage.

    spark-submit jobs/bronze_ingest.py --source-dir s3a://mimic-lake/raw
"""

from __future__ import annotations

import argparse
import sys

from lakehouse import dq
from lakehouse.config import LAKE
from lakehouse.io import read_csv, write_layer
from lakehouse.schemas import SOURCE_TABLES
from lakehouse.session import build_session, configure_logging

log = configure_logging()

# Row-count floors that distinguish "small demo dataset" from "the extract
# silently produced an empty file".
MIN_ROWS = {
    "patients": 10,
    "admissions": 10,
    "diagnoses_icd": 10,
    "d_labitems": 5,
    "labevents": 10,
    "chartevents": 0,
    "ecg_record_list": 0,
}


def suite_for(table: str) -> dq.Suite:
    suite = dq.Suite("bronze", table).expect_row_count_between(MIN_ROWS.get(table, 0))
    keys = {
        "patients": ["subject_id"],
        "admissions": ["subject_id", "hadm_id"],
        "diagnoses_icd": ["subject_id"],
        "d_labitems": ["itemid"],
        "labevents": ["subject_id", "itemid"],
        "ecg_record_list": ["subject_id", "study_id"],
    }.get(table, [])
    for column in keys:
        suite.expect_not_null(column, severity="error" if table != "labevents" else "warn")
    if table == "d_labitems":
        suite.expect_unique(["itemid"])
    return suite


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest raw MIMIC-IV CSVs into bronze Parquet")
    parser.add_argument(
        "--source-dir", default=f"{LAKE.root}/raw",
        help="Directory holding the raw MIMIC-IV CSV extracts",
    )
    parser.add_argument(
        "--tables", nargs="*", default=sorted(SOURCE_TABLES),
        help="Subset of tables to ingest (default: all)",
    )
    parser.add_argument(
        "--repartition", type=int, default=None,
        help="Target file count per table; keeps labevents from writing 1 huge file",
    )
    parser.add_argument("--skip-missing", action="store_true",
                        help="Tolerate absent optional sources (e.g. chartevents)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spark = build_session("mimic-bronze-ingest")
    failures: list[str] = []

    for table in args.tables:
        if table not in SOURCE_TABLES:
            raise SystemExit(f"unknown table '{table}'; known: {sorted(SOURCE_TABLES)}")
        relative_path, schema, partitions = SOURCE_TABLES[table]
        source = f"{args.source_dir.rstrip('/')}/{relative_path}"
        log.info("ingesting %s -> bronze.%s", source, table)
        try:
            frame = read_csv(spark, source, schema)
            suite_for(table).run(frame)
            write_layer(
                frame, LAKE.bronze_prefix, table,
                partition_by=partitions, repartition=args.repartition,
            )
        except Exception as exc:
            if args.skip_missing:
                log.warning("skipping %s: %s", table, exc)
                continue
            failures.append(f"{table}: {exc}")
            log.exception("failed to ingest %s", table)

    spark.stop()
    if failures:
        log.error("bronze ingest failed for %d table(s)", len(failures))
        return 1
    log.info("bronze ingest complete: %s", ", ".join(args.tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
