# Data quality

Quality checks run in two places, because they protect against two different
kinds of failure.

| Gate | Runs in | Catches | On failure |
| --- | --- | --- | --- |
| Spark expectations | inside each job, before the write | bad data entering a layer | `error` raises and fails the Airflow task; `warn` is recorded |
| dbt tests | after the warehouse load | broken assumptions between models | `dbt test` fails the DAG |

Every Spark check also writes a row to `_quality/dq_results` — run id,
timestamp, layer, table, check name, severity, observed value, threshold, row
count. That table is published to the warehouse and modelled as
`analytics_data_quality_history`, which turns "when did this start failing"
into a SQL query instead of a log grep.

## Severity

- **`error`** — the data is wrong in a way that invalidates everything
  downstream. Raise, fail the task, publish nothing.
- **`warn`** — suspicious but not disqualifying. Record it, keep going, and let
  `mimic_quality_monitor` surface the pattern over days rather than one run.

Getting this split right is the difference between a gate people trust and a
gate people disable. A `warn` that should be an `error` ships bad data; an
`error` that should be a `warn` gets commented out at 3am and never restored.

## Check catalogue — Spark

### bronze
| Check | Table | Severity | Protects against |
| --- | --- | --- | --- |
| row count ≥ floor | all | error | a silently truncated or empty extract |
| `not_null(subject_id)` | patients, admissions, diagnoses | error | an unjoinable partial extract |
| `unique(itemid)` | d_labitems | error | a duplicated dictionary inflating the lab join |

### silver
| Check | Table | Severity | Protects against |
| --- | --- | --- | --- |
| `unique(subject_id)` | patients | error | patient duplication multiplying every downstream join |
| `values_in(gender, [M, F])` | patients | warn | an unhandled source encoding |
| `between(anchor_age, 0, 120)` | patients | error | corrupt ages reaching the model as features |
| `unique(hadm_id)` | admissions | error | admission-grain violations |
| `between(los_hours, 0, 8760)` | admissions | error | inverted or corrupt timestamps |
| `unique(subject_id, hadm_id, icd_code, icd_version)` | diagnoses | error | duplicate diagnosis rows double-counting a patient |
| `labs_within_admission_window` | labevents | error | outpatient labs leaking in as stay features |

### gold
| Check | Table | Severity | Protects against |
| --- | --- | --- | --- |
| `unique(subject_id)` and `unique(hadm_id)` | cohort | error | the same patient in two splits |
| `values_in(label, [0, 1])` | cohort | error | a broken labelling rule |
| `values_in(split, …)` | cohort | error | a broken split assignment |
| `class_balance(label, 0.2, 0.8)` | cohort, mart | warn | a cohort that silently collapsed to one class |
| `no_patient_across_splits` | cohort | error | **leakage** — the failure that invalidates the study |
| `unique(hadm_id, itemid)` | lab_features | error | a broken pivot producing duplicate features |
| `values_in(aggregation, [first, mean])` | lab_features | error | the acute/stable rule not being applied |
| `null_rate(age) = 0` | feature_mart | error | a demographic join that half-missed |

## Check catalogue — dbt

**Schema tests** (in the `.yml` files): `not_null`, `unique`,
`accepted_values`, `relationships` between facts and dimensions, and
`dbt_utils.accepted_range` on every numeric with a physical bound.

**Source freshness**: `lake` sources warn at 26 hours and error at 72 — one
missed daily run is a warning, three is an incident.

**Singular tests** (`dbt/tests/`), the ones specific to this study:

| Test | What it protects |
| --- | --- |
| `assert_no_patient_across_splits` | the leakage invariant, re-checked at the warehouse boundary |
| `assert_cohort_class_balance` | a cohort that collapsed to one class still trains and still reports 95 % accuracy |
| `assert_every_split_has_both_classes` | AUC is undefined on a single-class split |
| `assert_lab_aggregation_rule_matches_marker_type` | acute markers really did use the first draw |
| `assert_first_value_within_measurement_range` | the promoted value lies inside the observed min/max |
| `assert_no_admission_outside_study_window` | corrupt timestamps that escaped silver |
| `assert_no_blocking_dq_failures_in_latest_run` | nobody published around the Spark gate |
| `assert_training_features_are_not_pre_imputed` | a well-meaning "fix" that fills NULLs and leaks test statistics |

That last one deserves a note: it flags any lab column with a **zero** null
rate. In a real clinical dataset every lab has missingness. A column with none
means someone imputed upstream — which is exactly the bug this whole
re-engineering was meant to eliminate.

## Adding a check

```python
from lakehouse import dq

(
    dq.Suite("gold", "cohort")
    .expect_unique(["subject_id"])
    .expect_class_balance("label", 0.2, 0.8, severity="warn")
    .expect_custom("my_rule", my_predicate)
    .run(dataframe)
)
```

`expect_custom` takes any `DataFrame -> CheckResult` callable, so a rule that
needs a window function or a self-join is a normal function, not a plugin.
