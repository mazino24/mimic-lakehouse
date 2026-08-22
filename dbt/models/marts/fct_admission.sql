{{ config(materialized='table', indexes=[{'columns': ['hadm_id'], 'unique': True}]) }}

-- Admission-grain fact table: one row per indexed hospital stay, with the lab
-- activity that happened during it.

with cohort as (
    select * from {{ ref('stg_cohort') }}
),

lab_activity as (
    select
        hadm_id,
        count(*)                                       as distinct_labs_measured,
        sum(measurement_count)                         as total_lab_measurements,
        count(*) filter (where is_acute_marker)        as acute_markers_measured,
        min(first_measurement_hours)                   as first_lab_hours_from_admit
    from {{ ref('stg_lab_features') }}
    group by hadm_id
)

select
    cohort.hadm_id,
    cohort.subject_id,
    cohort.label,
    cohort.cohort_group,
    cohort.split,
    cohort.admitted_at,
    cohort.discharged_at,
    cohort.length_of_stay_hours,
    round((cohort.length_of_stay_hours / 24.0)::numeric, 2) as length_of_stay_days,
    cohort.admit_year,
    cohort.admission_type,
    cohort.died_in_hospital,
    coalesce(lab_activity.distinct_labs_measured, 0)        as distinct_labs_measured,
    coalesce(lab_activity.total_lab_measurements, 0)        as total_lab_measurements,
    coalesce(lab_activity.acute_markers_measured, 0)        as acute_markers_measured,
    lab_activity.first_lab_hours_from_admit,
    -- "Was any lab drawn at all" is itself signal: an untested patient is a
    -- different clinical situation from a tested one with normal results.
    (lab_activity.hadm_id is not null)                      as has_any_lab,
    cohort.etl_run_id
from cohort
left join lab_activity on lab_activity.hadm_id = cohort.hadm_id
