#!/usr/bin/env python3
"""Job 4 — gold Parquet -> Postgres warehouse.

Spark writes into the ``lake`` schema; dbt then builds ``staging`` ->
``intermediate`` -> ``marts`` on top. Keeping the load dumb (no transformation
in the JDBC step) means the modelling layer is version-controlled SQL rather
than Python nobody can review.

    spark-submit --jars /opt/jars/postgresql.jar jobs/publish_warehouse.py
"""

from __future__ import annotations

import argparse
import sys

from lakehouse.config import LAKE, WAREHOUSE
from lakehouse.io import drop_lineage, read_layer, read_parquet, write_jdbc
from lakehouse.session import build_session, configure_logging

log = configure_logging()

#: gold table -> warehouse table
PUBLISH = {
    "cohort": "cohort",
    "lab_features": "lab_features",
    "feature_mart": "feature_mart",
    "feature_coverage": "feature_coverage",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish gold tables to Postgres")
    parser.add_argument("--tables", nargs="*", default=sorted(PUBLISH))
    parser.add_argument("--mode", default="overwrite", choices=["overwrite", "append"])
    parser.add_argument("--skip-dq", action="store_true", help="Do not publish dq_results")
    args = parser.parse_args(argv)

    spark = build_session("mimic-publish-warehouse")
    log.info("publishing to %s (schema %s)", WAREHOUSE.jdbc_url, WAREHOUSE.schema)

    for table in args.tables:
        target = PUBLISH[table]
        # `_loaded_at` must survive: dbt source freshness is configured
        # against it, and it is how a warehouse row is traced to its run.
        frame = drop_lineage(
            read_layer(spark, LAKE.gold_prefix, table), keep=["_run_id", "_loaded_at"]
        )
        write_jdbc(frame, target, mode=args.mode)

    if not args.skip_dq:
        try:
            results = read_parquet(spark, LAKE.quality("dq_results"))
            # DQ history is append-only: it is a time series, not a snapshot.
            write_jdbc(results, "dq_results", mode="append")
        except Exception as exc:
            log.warning("no dq_results to publish: %s", exc)

    spark.stop()
    log.info("warehouse publish complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
