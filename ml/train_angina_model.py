#!/usr/bin/env python3
"""Downstream consumer: train angina classifiers on the warehouse mart.

This is deliberately thin. The platform's job is to hand the model a table it
can trust; this script's job is to read that table, respect the split the
platform assigned, and write its metrics back so model quality lives next to
data quality in the same warehouse.

    python ml/train_angina_model.py --source warehouse --run-id 2026-08-23
    python ml/train_angina_model.py --source parquet --path data/gold/feature_mart
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("train")

TARGET = "label"
NON_FEATURES = {
    "hadm_id", "subject_id", "label", "split", "etl_run_id", "_run_id",
    "_loaded_at", "_source", "split_bucket", "ecg_path", "ecg_study_id",
    "admittime", "dischtime", "gender", "cohort_group", "feature_count",
}


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


def load_features(source: str, path: str | None, table: str) -> pd.DataFrame:
    if source == "warehouse":
        from sqlalchemy import create_engine

        engine = create_engine(warehouse_url())
        log.info("reading %s from the warehouse", table)
        return pd.read_sql(f"select * from {table}", engine)
    log.info("reading parquet from %s", path)
    return pd.read_parquet(path)


def build_models(seed: int) -> dict[str, Pipeline]:
    """Every sklearn model is wrapped in a Pipeline.

    The imputer and scaler are fit inside the pipeline, which means they are
    fit on training rows only. Imputing before the split is the single most
    common way a medical ML result turns out to be too good to be true.
    """
    numeric = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [*numeric, ("model", LogisticRegression(max_iter=2000, random_state=seed))]
        ),
        "random_forest": Pipeline([
            *numeric,
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed
                ),
            ),
        ]),
        "gradient_boosting": Pipeline(
            [*numeric, ("model", GradientBoostingClassifier(random_state=seed))]
        ),
    }
    try:
        from xgboost import XGBClassifier

        # XGBoost handles NaN natively, so it gets the raw matrix: the absence
        # of a lab result is information, not something to average away.
        models["xgboost"] = Pipeline([(
            "model",
            XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=seed,
            ),
        )])
    except ImportError:
        log.warning("xgboost not installed; skipping that model")
    return models


def evaluate(model, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    both_classes = len(np.unique(y)) > 1
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y, probabilities) if both_classes else float("nan"),
    }


def write_metrics(rows: list[dict], run_id: str) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(warehouse_url())
    frame = pd.DataFrame(rows)
    with engine.begin() as connection:
        # Re-running a date must replace that run, not accumulate duplicates.
        connection.execute(
            text("delete from lake.model_metrics where run_id = :run_id"), {"run_id": run_id}
        )
        frame.to_sql("model_metrics", connection, schema="lake", if_exists="append", index=False)
    log.info("wrote %d metric rows to lake.model_metrics", len(frame))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=["warehouse", "parquet"], default="warehouse")
    parser.add_argument("--path", default=None, help="Parquet path when --source parquet")
    parser.add_argument("--table", default="marts.mart_angina_training_features")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-warehouse-write", action="store_true")
    args = parser.parse_args()

    frame = load_features(args.source, args.path, args.table)
    if TARGET not in frame.columns:
        raise SystemExit(f"no '{TARGET}' column in {args.table}")

    # Booleans (has_ecg, troponin_elevated) are usable features once cast;
    # everything non-numeric and non-boolean is metadata, not signal.
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype("float64")

    feature_columns = [
        column
        for column in frame.columns
        if column not in NON_FEATURES and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_columns:
        raise SystemExit("no numeric feature columns found in the mart")

    # The split was decided by the platform, by patient hash. Training must not
    # re-split: that is how the same patient ends up on both sides.
    splits = {
        name: frame[frame["split"] == name] for name in ("train", "validation", "test")
    }
    if splits["train"].empty or splits["test"].empty:
        raise SystemExit("train or test split is empty — check the cohort build")

    log.info(
        "loaded %s rows, %d features (train=%d validation=%d test=%d)",
        f"{len(frame):,}", len(feature_columns), *(len(s) for s in splits.values()),
    )

    X = {name: part[feature_columns] for name, part in splits.items()}
    y = {name: part[TARGET].astype(int).to_numpy() for name, part in splits.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    report: dict[str, dict] = {}

    for name, model in build_models(args.seed).items():
        log.info("training %s", name)
        model.fit(X["train"], y["train"])
        report[name] = {}
        for split_name in ("validation", "test"):
            if splits[split_name].empty:
                continue
            scores = evaluate(model, X[split_name], y[split_name])
            report[name][split_name] = scores
            metric_rows += [
                {
                    "run_id": args.run_id,
                    "trained_at": datetime.now(timezone.utc),
                    "model_name": name,
                    "split": split_name,
                    "metric_name": metric,
                    "metric_value": None if pd.isna(value) else float(value),
                    "n_rows": len(splits[split_name]),
                    "n_features": len(feature_columns),
                }
                for metric, value in scores.items()
            ]
        test_scores = report[name].get("test", {})
        log.info(
            "  %-20s test ROC-AUC=%.4f  F1=%.4f  recall=%.4f",
            name, test_scores.get("roc_auc", float("nan")),
            test_scores.get("f1", float("nan")), test_scores.get("recall", float("nan")),
        )

    best = max(
        report, key=lambda name: report[name].get("test", {}).get("roc_auc", 0) or 0
    )
    summary = {
        "run_id": args.run_id,
        "rows": len(frame),
        "features": len(feature_columns),
        "feature_columns": feature_columns,
        "best_model": best,
        "metrics": report,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("best model: %s (artifacts in %s)", best, args.output_dir)

    if not args.skip_warehouse_write:
        write_metrics(metric_rows, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
