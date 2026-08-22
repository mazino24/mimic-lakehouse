{{ config(materialized='view') }}

-- Light renaming only: staging never changes grain or filters rows, so the
-- warehouse always has an untouched view of what Spark published.

with source as (
    select * from {{ source('lake', 'cohort') }}
)

select
    subject_id,
    hadm_id,
    label,
    case when label = 1 then 'angina' else 'control' end as cohort_group,
    gender,
    case when gender = 'M' then 1 else 0 end             as is_male,
    anchor_age                                            as age_years,
    width_bucket(anchor_age, 18, 98, 8)                   as age_bucket,
    admittime                                             as admitted_at,
    dischtime                                             as discharged_at,
    los_hours                                             as length_of_stay_hours,
    admit_year,
    admission_type,
    insurance,
    race,
    died_in_hospital,
    split,
    _run_id                                               as etl_run_id
from source
