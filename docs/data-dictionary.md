# Data dictionary

Lineage columns (`_loaded_at`, `_run_id`, `_source`) are stamped on every lake
table and omitted from the tables below.

## Sources — MIMIC-IV `hosp` module

| File | Rows (real extract) | Used for |
| --- | --- | --- |
| `patients.csv` | ~365k | demographics, adult filter |
| `admissions.csv` | ~546k | stay windows, length of stay, disposition |
| `diagnoses_icd.csv` | ~6.4M | cohort labelling (ICD-10) |
| `d_labitems.csv` | ~1.6k | lab dictionary, target/acute classification |
| `labevents.csv` | ~130M (41 GB) | lab features |
| `chartevents.csv` | ~330M | reserved for vitals (ingested, not yet modelled) |
| `mimic-iv-ecg/record_list.csv` | ~800k | ECG study index |

## silver

### `silver.patients` — one row per patient
| Column | Type | Notes |
| --- | --- | --- |
| `subject_id` | int | PK |
| `gender` | string | normalised to `M` / `F`, else NULL |
| `anchor_age` | int | NULL outside [0, 120]; MIMIC shifts ages > 89 to 91 |
| `is_adult` | boolean | `anchor_age >= 18` |
| `anchor_year`, `anchor_year_group`, `dod` | | as sourced |

### `silver.admissions` — one row per stay
| Column | Type | Notes |
| --- | --- | --- |
| `hadm_id` | int | PK |
| `subject_id` | int | FK to patients |
| `admittime`, `dischtime` | timestamp | stays with `dischtime < admittime` are dropped |
| `los_hours` | double | length of stay |
| `admit_year` | int | **partition key** |
| `died_in_hospital` | boolean | from `hospital_expire_flag` |
| `admission_type`, `insurance`, `race`, `marital_status` | string | as sourced |

### `silver.diagnoses` — one row per (stay, code)
| Column | Type | Notes |
| --- | --- | --- |
| `icd_code` | string | uppercased, dots stripped: `I25.110` → `I25110` |
| `is_angina` | boolean | ICD-10 prefix `I20` or `I251` |
| `is_circulatory` | boolean | any ICD-10 `I*` — the control exclusion rule |
| `is_primary` | boolean | `seq_num = 1` |

### `silver.lab_dictionary`
| Column | Type | Notes |
| --- | --- | --- |
| `itemid` | int | PK |
| `label` | string | free text, e.g. `"Cholesterol, LDL, Calculated"` |
| `is_target_lab` | boolean | matches a configured panel pattern |
| `is_acute_marker` | boolean | Troponin or CK-MB — drives first-vs-mean |

### `silver.labevents` — one row per in-window measurement
| Column | Type | Notes |
| --- | --- | --- |
| `hadm_id`, `itemid` | int | |
| `charttime` | timestamp | guaranteed within the admission window |
| `hours_from_admit` | double | ≥ 0 by construction |
| `valuenum` | double | non-NULL by construction; text results are dropped |
| `marker_type` | string | **partition key** — `acute` / `routine` |
| `is_acute_marker` | boolean | same fact as a boolean, for downstream logic |

### `silver.ecg_records` — one ECG per stay
| Column | Type | Notes |
| --- | --- | --- |
| `hadm_id` | int | PK — earliest in-window study wins |
| `study_id`, `path` | | pointer into the PhysioNet waveform tree |
| `hours_from_admit` | double | ≤ 72; a 12-hour pre-admission grace covers ED ECGs |

## gold

### `gold.cohort` — one row per patient
| Column | Type | Notes |
| --- | --- | --- |
| `subject_id` | int | PK — a patient appears exactly once |
| `hadm_id` | int | the indexed stay (their first eligible admission) |
| `label` | int | 1 = angina case, 0 = control |
| `split` | string | **partition key** — `train` / `validation` / `test` |
| `split_bucket` | int | 0–99, the hash bucket behind `split` |
| `gender`, `anchor_age`, `admittime`, `dischtime`, `los_hours` | | |

### `gold.lab_features` — one row per (stay, lab)
| Column | Type | Notes |
| --- | --- | --- |
| `feature_value` | double | **the value promoted to a feature** |
| `aggregation` | string | `first` for acute markers, `mean` otherwise |
| `first_value`, `mean_value`, `min_value`, `max_value` | double | full distribution retained |
| `measurement_count` | int | how often it was ordered — itself signal |
| `first_measurement_hours` | double | time to first draw |

### `gold.feature_mart` — the modelling table
One row per stay: cohort columns, one column per surviving lab (labs measured
in < 2 % of admissions are dropped), `gender_male`, `age`, `has_ecg`, and the
ECG pointer. **NULLs are preserved deliberately.**

### `gold.feature_coverage`
| Column | Type | Notes |
| --- | --- | --- |
| `column_name` | string | |
| `non_null_rows`, `total_rows` | bigint | |
| `null_rate` | double | drift in this is the earliest warning of an upstream break |

### `_quality.dq_results`
`run_id`, `checked_at`, `layer`, `table_name`, `check_name`, `severity`,
`passed`, `observed`, `threshold`, `row_count`, `details`. Append-only.

## Warehouse marts

| Model | Grain | Purpose |
| --- | --- | --- |
| `dim_patient` | patient | demographics with age bands |
| `fct_admission` | stay | stay facts + lab activity counts |
| `fct_lab_result` | stay × lab | every promoted lab value with its distribution |
| `mart_angina_training_features` | patient | **the ML contract** |
| `analytics_cohort_summary` | group × split | population characteristics |
| `analytics_lab_profile_by_cohort` | lab | case-vs-control means with Cohen's *d* |
| `analytics_feature_coverage` | column | completeness tiers |
| `analytics_data_quality_history` | run × table | pass rates over time |
| `analytics_model_performance` | run × model × split | metrics with run-over-run deltas |
