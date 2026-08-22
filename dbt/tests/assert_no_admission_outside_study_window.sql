-- Length of stay must be non-negative and shorter than a year; anything else
-- is a corrupt timestamp that escaped the silver layer.

select
    hadm_id,
    admitted_at,
    discharged_at,
    length_of_stay_hours
from {{ ref('stg_cohort') }}
where length_of_stay_hours < 0
   or length_of_stay_hours > {{ var('max_lab_hours_from_admit') }}
   or discharged_at < admitted_at
