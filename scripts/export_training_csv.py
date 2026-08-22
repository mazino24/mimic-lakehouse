#!/usr/bin/env python3
"""Export the warehouse mart to CSV/Parquet for the deep-learning consumers.

The multimodal models in ``ml/multimodal`` were written against a flat CSV
before the lakehouse existed. Rather than rewrite them, the platform hands
them the file shape they expect — which is what a data platform should do for
any consumer it does not own.

    python scripts/export_training_csv.py --out data/processed/tabular_features.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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
    parser.add_argument("--table", default="marts.mart_angina_training_features")
    parser.add_argument("--out", type=Path, default=Path("data/processed/tabular_features.csv"))
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    args = parser.parse_args()

    import pandas as pd
    from sqlalchemy import create_engine

    frame = pd.read_sql(f"select * from {args.table}", create_engine(warehouse_url()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        frame.to_csv(args.out, index=False)
    else:
        frame.to_parquet(args.out, index=False)

    print(f"exported {len(frame):,} rows x {len(frame.columns)} columns -> {args.out}")
    print(f"  label balance: {frame['label'].mean():.3f}")
    print(f"  splits: {frame['split'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
