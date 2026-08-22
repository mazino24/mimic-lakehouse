{{ config(materialized='table', indexes=[{'columns': ['subject_id'], 'unique': True}]) }}

-- One row per patient in the study population.

select
    subject_id,
    gender,
    is_male,
    age_years,
    age_bucket,
    case
        when age_years < 45 then 'under_45'
        when age_years < 65 then '45_64'
        when age_years < 80 then '65_79'
        else '80_plus'
    end                                           as age_group,
    race,
    insurance,
    label                                         as has_angina,
    cohort_group,
    split,
    etl_run_id
from {{ ref('stg_cohort') }}
