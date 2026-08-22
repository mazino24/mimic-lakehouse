{{ config(materialized='table', indexes=[{'columns': ['hadm_id']}, {'columns': ['lab_name']}]) }}

-- Lab-grain fact: one row per (admission, lab test) with the value that was
-- promoted to a feature and the raw distribution it came from.

select
    {{ dbt_utils.generate_surrogate_key(['hadm_id', 'lab_itemid']) }} as lab_result_key,
    hadm_id,
    subject_id,
    lab_itemid,
    lab_name,
    is_acute_marker,
    aggregation_rule,
    feature_value,
    measurement_count,
    first_measurement_hours,
    label,
    cohort_group,
    split
from {{ ref('int_cohort_lab_panel') }}
