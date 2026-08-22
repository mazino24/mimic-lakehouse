# MIMIC-IV Angina Lakehouse

A production-shaped data platform that turns **41 GB of raw hospital records**
into a governed, tested feature warehouse — and a clinical ML model as its
first consumer.

Built on the data behind my master's thesis (angina pectoris prediction from
MIMIC-IV, Astana IT University), re-engineered from a folder of pandas scripts
into an orchestrated lakehouse: **PySpark → MinIO/S3 → Postgres → dbt → Airflow**,
with data-quality gates at every boundary and CI that builds the whole thing
from scratch on every push.

```
Raw CSV (41 GB)  →  bronze  →  silver  →  gold  →  warehouse  →  marts  →  model
                     typed    conformed  labelled   Postgres     dbt      XGBoost
                    Parquet   + clinical  cohort +   (JDBC)     tested    + CNN
                              annotation  features
```

---

## Why this exists

The research produced good results and an unmaintainable pipeline: 14 scripts
run by hand in an order that only I remembered, imputation applied before the
train/test split, lab values averaged across measurements taken months apart,
and no way to tell whether a rerun would reproduce yesterday's numbers.

Every one of those is a data engineering problem, so this repo fixes them as
data engineering problems — in the pipeline, enforced by tests, not in a
notebook cell that someone has to remember to run.

---

## Architecture

```mermaid
flowchart LR
    subgraph src["Sources"]
        A1["MIMIC-IV<br/>hosp module<br/>41 GB CSV"]
        A2["MIMIC-IV-ECG<br/>800k studies<br/>2.8 GB WFDB"]
    end

    subgraph lake["Object storage · MinIO / S3"]
        B["bronze<br/>typed Parquet<br/>+ lineage"]
        C["silver<br/>conformed entities<br/>clinical flags"]
        D["gold<br/>cohort · features<br/>coverage"]
        Q["_quality<br/>dq_results<br/>history"]
    end

    subgraph wh["Warehouse · Postgres"]
        E["lake<br/>published gold"]
        F["staging → intermediate<br/>dbt views"]
        G["marts<br/>dim / fct / feature contract"]
        H["analytics<br/>cohort · labs · DQ · model perf"]
    end

    subgraph cons["Consumers"]
        I["XGBoost / GBM<br/>tabular baseline"]
        J["1D-CNN fusion<br/>tabular + ECG"]
        K["dbt docs<br/>+ analytics SQL"]
    end

    A1 --> B
    A2 --> B
    B -->|PySpark| C
    C -->|PySpark| D
    D -->|JDBC| E
    E --> F --> G --> H
    G --> I
    G --> J
    H --> K
    B -.-> Q
    C -.-> Q
    D -.-> Q
    Q --> E

    AF(["Airflow<br/>3 DAGs · dataset-triggered"]) -.orchestrates.-> B
    AF -.-> C
    AF -.-> D
    AF -.-> E
    AF -.-> F
    AF -.-> I
```

**Stack:** PySpark 3.5 · Airflow 2.10 · dbt 1.8 · PostgreSQL 16 · MinIO (S3A) ·
Docker Compose · GitHub Actions · pytest · scikit-learn / XGBoost / PyTorch

---

## Quickstart

Requires Docker and ~8 GB of free RAM. **No PhysioNet credentials needed** —
the repo ships a synthetic data generator that reproduces MIMIC-IV's schemas
*and its quirks*.

```bash
git clone <this-repo> && cd mimic-lakehouse
make demo
```

`make demo` builds the images, starts the stack, generates a synthetic extract,
uploads it to MinIO, and triggers the pipeline.

| Service | URL | Credentials |
| --- | --- | --- |
| Airflow | http://localhost:8080 | `admin` / `admin` |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Spark master | http://localhost:8081 | — |
| Warehouse | `postgresql://mimic:mimic@localhost:5433/mimic` | — |

Run the tests without Docker at all:

```bash
make venv && make test     # 32 tests, including a full bronze→gold build
```

---

## What the pipeline actually does

### 1. Bronze — typed landing zone
Raw CSVs read with **explicit schemas** (never `inferSchema`: it costs a full
extra pass over 40 GB and silently retypes `hadm_id` to double the first time a
null appears), written as Snappy Parquet with lineage columns stamping the run
that produced every row.

### 2. Silver — conformed entities
Deduplication to a stable grain, gender/age normalisation, ICD-10 codes
stripped of dots and flagged (`is_angina`, `is_circulatory`), and the pipeline's
heaviest join: **40 GB of lab events narrowed to in-window measurements** via a
broadcast dictionary join plus an admission-window filter.

### 3. Gold — the labelled study cohort
Cases, controls, features, and the split — all decided once, in one place:

- **cases**: adult stays with ICD-10 `I20*` / `I251*`
- **controls**: adults with *no circulatory diagnosis anywhere in their record*
- **one stay per patient**, so nobody can appear on both sides of a split
- **split assigned by hash of `subject_id`** — deterministic across reruns and
  machines
- **acute markers aggregated by first draw, stable markers by mean** (below)
- **NULLs preserved**: a test that was never ordered is clinical signal

### 4. Warehouse + dbt
Gold lands in Postgres via JDBC; dbt builds staging → intermediate → marts →
analytics. **31 dbt tests** — schema tests plus 8 singular tests that encode
research invariants (no patient across splits, class balance intact, the
aggregation rule survived the trip through Parquet and JDBC).

### 5. Consumers
`mart_angina_training_features` is the contract. The tabular models and the
multimodal ECG network both read it; neither re-splits, neither imputes.
Metrics are written back to `lake.model_metrics`, so **model quality sits next
to the data-quality history that explains it**.

---

## Engineering decisions worth defending

**Acute markers use the first in-stay draw, not the mean.**
Troponin and CK-MB are diagnostic at presentation. Averaging serial draws
taken after treatment started dilutes exactly the signal being measured. The
rule is declared in config, applied in Spark, and asserted by both a pytest
test and a dbt test.

**Controls exclude every patient with any circulatory diagnosis.**
Heart-failure and MI patients share the angina lab signature. Leaving them in
teaches a model "cardiac vs non-cardiac" while you believe it learned
"angina vs not".

**The split is a property of the data, not of the training script.**
Assigned upstream by patient hash, materialised in the mart, tested at the
warehouse boundary. A training script *cannot* leak patients across splits
because it never gets to choose.

**Missing values stay missing until the training fold.**
The original pipeline filled NaNs with the population median over the full
dataset — test statistics bleeding into training, plus the destruction of
missingness as a feature. Imputation now lives inside sklearn `Pipeline`s, fit
per fold. XGBoost gets the raw matrix.

**Data quality is a time series, not a log line.**
Every check writes a row to `_quality/dq_results` with severity, observed
value and threshold. `error` fails the Airflow task; `warn` is recorded and
surfaces in the monitoring DAG. `analytics_data_quality_history` makes
"when did this table start degrading" a SQL query.

**Full-refresh jobs use STATIC overwrite.**
Found the hard way: with `partitionOverwriteMode=dynamic`, changing a
partition key leaves orphaned directories and the next read dies on
conflicting schemas. There is a test for that now.

---

## Results

Trained on the synthetic demo extract (3,000 patients → 2,714-patient cohort,
36 features), all metrics on the held-out test split:

| Model | ROC-AUC | F1 | Recall |
| --- | --- | --- | --- |
| XGBoost | 0.865 | 0.772 | 0.766 |
| Logistic regression | 0.862 | 0.775 | 0.781 |
| Random forest | 0.857 | 0.776 | 0.776 |
| Gradient boosting | 0.857 | 0.754 | 0.746 |

On the **real MIMIC-IV extract** (58,486 admissions, 34 features) the same
downstream models reach **ROC-AUC 0.797 / F1 0.742** — the thesis result. The
synthetic numbers run a little higher because the generated signal is cleaner
than reality; that is what synthetic data is for.

---

## Repository layout

```
spark/
  lakehouse/            reusable library — config, IO, schemas, DQ engine
    transforms/         pure DataFrame→DataFrame logic (the business rules)
  jobs/                 4 spark-submit entrypoints: bronze, silver, gold, publish
  tests/                32 pytest tests incl. a full end-to-end lake build
airflow/dags/           3 DAGs: ELT, dataset-triggered training, quality monitor
dbt/
  models/               staging → intermediate → marts → analytics
  tests/                8 singular tests encoding research invariants
ml/                     downstream consumers + the regression gate
  multimodal/           thesis-era ECG fusion models (tabular + 1D-CNN)
scripts/                synthetic data generator, lake seeding, mart export
docker/                 Spark and Airflow images (S3A + JDBC jars)
warehouse/init/         warehouse schema bootstrap
docs/                   architecture, data dictionary, runbook
```

---

## Running against real MIMIC-IV

The pipeline is schema-compatible with the real extract; only the input path
changes.

1. Complete CITI training and get credentialed access to
   [MIMIC-IV](https://physionet.org/content/mimiciv/) on PhysioNet.
2. Download the `hosp` module and drop `patients.csv`, `admissions.csv`,
   `diagnoses_icd.csv`, `d_labitems.csv`, `labevents.csv` into `data/raw/`.
3. Optional ECG modality: fetch `mimic-iv-ecg/record_list.csv`, then use
   `silver.ecg_records` to download only the studies the cohort needs
   (~2.8 GB instead of ~800 GB).
4. `make seed && make pipeline`.

Scale the cluster first — `SPARK_WORKERS`, `SPARK_WORKER_MEMORY` and
`SPARK_SHUFFLE_PARTITIONS` in `.env`. At 40 GB, set shuffle partitions to
roughly 3× total executor cores.

**No MIMIC-IV data is committed to this repository, and none should be.** The
PhysioNet data use agreement prohibits redistribution; `.gitignore` blocks
`data/` and every data file extension for that reason.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — layer contracts, orchestration, scaling notes
- [`docs/data-dictionary.md`](docs/data-dictionary.md) — every table and column
- [`docs/data-quality.md`](docs/data-quality.md) — the check catalogue and what each protects
- [`docs/runbook.md`](docs/runbook.md) — operating the pipeline, failure playbooks
- [`docs/cv.md`](docs/cv.md) — what this project demonstrates, in resume form

---

## License & attribution

Code: MIT. Data: MIMIC-IV is credentialed and governed by the PhysioNet DUA —
not included here. Thesis supervised by Kuatbayeva Akmaral Alikhanova,
Astana IT University.
