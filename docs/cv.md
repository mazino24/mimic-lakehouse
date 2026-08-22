# CV material

Copy-paste sources for a resume, LinkedIn, or an interview. Numbers are from
the repo as built — keep them accurate as it changes.

## One-line summary

> Built a production-shaped clinical data platform (PySpark, Airflow, dbt,
> Postgres, MinIO/S3, Docker) that processes 41 GB of MIMIC-IV hospital
> records into a tested feature warehouse, with data-quality gates at every
> layer boundary and CI that rebuilds the entire stack on each push.

## Resume bullets

Pick three or four; leading with an outcome beats leading with a tool.

- Designed and built an end-to-end **medallion lakehouse** (bronze → silver →
  gold) over **41 GB** of MIMIC-IV hospital data using **PySpark on a
  standalone cluster with MinIO/S3 object storage**, replacing 14 ad-hoc pandas
  scripts with an orchestrated, reproducible pipeline.
- Orchestrated the platform with **Airflow**: 3 DAGs, 18 tasks, per-stage
  retries with exponential backoff, and **dataset-triggered** downstream
  training so model runs follow data landing rather than a cron guess.
- Implemented a **data-quality framework** running 45+ declarative expectations per pipeline run
  inside the Spark jobs; blocking failures fail the DAG, and every
  check is persisted as a time series that surfaces as a warehouse table.
- Modelled the warehouse in **dbt** — 15 models across staging / intermediate /
  marts / analytics, validated by **58 dbt tests** including 8 custom SQL tests that
  encode research invariants (no patient across train/test splits, class
  balance, aggregation-rule integrity).
- Eliminated **train/test leakage** by moving split assignment upstream into the
  pipeline (deterministic hash by patient) and keeping imputation inside
  per-fold sklearn pipelines — a documented defect in the original research
  code, now prevented by tests at two layers.
- Optimised the pipeline's heaviest stage (a **130M-row lab-event join**) with
  broadcast joins, predicate pushdown, explicit schemas and adaptive query
  execution with skew handling.
- Shipped the project as a **one-command Docker Compose stack** with a
  synthetic data generator, so the platform runs end to end without
  credentialed access to the source data; **GitHub Actions** rebuilds the lake,
  the warehouse and the models on every push.
- Delivered the ML workload as a **downstream consumer of a governed contract
  table**, writing metrics back into the warehouse and gating deployments on a
  run-over-run regression check (test ROC-AUC 0.797 on the real extract).

## Project entry (portfolio / LinkedIn)

**MIMIC-IV Angina Lakehouse** — *Data platform engineering*
PySpark · Airflow · dbt · PostgreSQL · MinIO/S3 · Docker · GitHub Actions · pytest

Re-engineered my master's thesis pipeline into a production-shaped data
platform. Raw hospital records (41 GB across 7 tables, plus 800k ECG studies)
flow through a medallion lakehouse into a dbt-modelled Postgres warehouse, with
quality gates at every boundary and a clinical ML model as the first consumer.
The interesting problems were not the models: they were leakage, aggregation
correctness, reproducibility, and making a research pipeline something a second
person could run.

## Talking points for interviews

**"Tell me about a data quality problem you found."**
Lab values were being averaged across every measurement in a stay. For
troponin — the marker that diagnoses cardiac events — that is wrong: it is
drawn at presentation, and averaging in later draws taken after treatment
started dilutes the signal. I split aggregation by marker type (first draw for
acute markers, mean for stable ones), encoded it in config, and asserted it in
both a pytest test and a dbt test so it cannot silently regress.

**"Tell me about a bug that would have been expensive."**
The original pipeline imputed missing labs with the population median across
the full dataset before splitting — test-set statistics leaking into training,
plus the destruction of missingness, which in medicine is itself signal (a test
that was never ordered tells you something). The fix was structural: the
warehouse keeps NULLs, imputation happens inside per-fold sklearn pipelines,
and a dbt test fails the build if any lab column ever shows a zero null rate.

**"How do you prevent leakage?"**
By taking the decision away from the training script. Split assignment is a
hash of the patient id, applied once in the pipeline and materialised in the
mart. Training reads the split; it cannot choose one. Two independent tests
assert that no patient appears in more than one split — one in Spark, one in
dbt at the warehouse boundary.

**"What would you do differently at 10x the scale?"**
Move from bare Parquet to Iceberg or Delta for ACID and schema evolution, make
the lab-event load incremental instead of a full refresh (the lineage columns
are already in place for it), and add OpenLineage for column-level lineage. The
orchestration and modelling layers would not need to change, which is the point
of the layer contracts.

**"What is the hardest thing you debugged in this project?"**
A boolean column that came back as a string. I had partitioned the silver lab
table by `is_acute_marker`; Hive-style partition values round-trip as text, so
every downstream `when(col(...))` broke with a type error. The fix was a string
partition key with the boolean kept as a regular column — and an end-to-end
test that asserts the round-trip type, because unit tests on in-memory
DataFrames could never have caught it.

## Honest framing

If asked what is real and what is demo:

- The pipeline, tests, orchestration and CI are real and run.
- The **data** in the public demo is synthetic — MIMIC-IV is credentialed and
  cannot be redistributed. The generator reproduces the real schemas and their
  quirks; the pipeline is schema-compatible with the genuine extract, which is
  what produced the 0.797 ROC-AUC figure.
- The Spark cluster is two Docker workers, not a hundred-node YARN cluster. The
  code path is identical; the sizing is not.

Saying this before you are asked reads as engineering maturity. Claiming a
hundred-node cluster does not.
