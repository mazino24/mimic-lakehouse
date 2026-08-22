#!/usr/bin/env python3
"""Fail the DAG when model quality drops without an obvious data cause.

Compares the newest run's metric against the median of the previous runs. A
drop bigger than the tolerance is treated as an incident, because in practice
it almost always traces back to the pipeline (a column that went null, a
cohort that shifted) rather than to the model code, which did not change.

    python ml/check_regression.py --run-id 2026-08-23 --metric roc_auc --max-drop 0.03
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("regression-gate")


def warehouse_url() -> str:
    """Warehouse DSN.

    ``WAREHOUSE_URL`` wins when set, so a caller can point at any instance
    (a socket, a managed cloud database, a CI service) without having to
    decompose it into five variables.
    """
    explicit = os.getenv("WAREHOUSE_URL")
    if explicit:
        return explicit
    return (
        f"postgresql+psycopg2://{os.getenv('WAREHOUSE_USER', 'mimic')}:"
        f"{os.getenv('WAREHOUSE_PASSWORD', 'mimic')}@"
        f"{os.getenv('WAREHOUSE_HOST', 'localhost')}:"
        f"{os.getenv('WAREHOUSE_PORT', '5433')}/"
        f"{os.getenv('WAREHOUSE_DB', 'mimic')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-drop", type=float, default=0.03)
    parser.add_argument("--min-history", type=int, default=2)
    args = parser.parse_args()

    import pandas as pd
    from sqlalchemy import create_engine

    engine = create_engine(warehouse_url())
    history = pd.read_sql(
        """
        select run_id, trained_at, model_name, metric_value
        from lake.model_metrics
        where metric_name = %(metric)s and split = %(split)s
        order by trained_at
        """,
        engine,
        params={"metric": args.metric, "split": args.split},
    )
    if history.empty:
        log.warning("no metric history yet — nothing to compare against")
        return 0

    current = history[history["run_id"] == args.run_id]
    previous = history[history["run_id"] != args.run_id]
    if current.empty:
        log.error("run %s wrote no %s metric", args.run_id, args.metric)
        return 1
    if len(previous["run_id"].unique()) < args.min_history:
        log.info("only %d prior run(s); baseline not established yet",
                 len(previous["run_id"].unique()))
        return 0

    current_best = current["metric_value"].max()
    baseline = previous.groupby("run_id")["metric_value"].max().median()
    drop = baseline - current_best

    log.info("%s: current=%.4f baseline(median of %d runs)=%.4f drop=%.4f",
             args.metric, current_best, previous["run_id"].nunique(), baseline, drop)

    if drop > args.max_drop:
        log.error(
            "REGRESSION: %s fell %.4f below baseline (tolerance %.4f). "
            "Check analytics_feature_coverage and analytics_data_quality_history "
            "for the same run before blaming the model.",
            args.metric, drop, args.max_drop,
        )
        return 1
    log.info("model quality within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
