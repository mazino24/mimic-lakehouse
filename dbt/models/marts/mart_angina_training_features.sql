{{ config(
    materialized='table',
    indexes=[{'columns': ['hadm_id'], 'unique': True}, {'columns': ['split']}]
) }}

{#
  The contract between the data platform and the models.

  Two rules make this table safe to train on:
    1. rows are one-per-patient and `split` is assigned by patient hash
       upstream, so nothing leaks between train and test;
    2. missing values stay NULL. Imputation belongs inside the training fold,
       never in a shared table that both train and test read from.
#}

{% set mart = ref('stg_feature_mart') %}

with base as (
    select * from {{ mart }}
)

select
    hadm_id,
    subject_id,
    label,
    split,
    age,
    gender_male,
    los_hours,
    has_ecg,
    {% if column_exists(mart, 'troponin_t') -%}
    -- Clinically meaningful thresholds, computed once here rather than in
    -- every downstream notebook.
    (troponin_t > 0.10)                        as troponin_elevated,
    {% else -%}
    cast(null as boolean)                      as troponin_elevated,
    {% endif -%}
    {% for column in lab_feature_columns(mart) -%}
    {{ adapter.quote(column) }},
    {% endfor -%}
    _run_id                                    as etl_run_id
from base
