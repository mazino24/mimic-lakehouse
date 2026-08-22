-- Long-form lab values joined to the patient's label. Ephemeral: it exists to
-- keep the two analytics models below readable, not to be queried directly.

with cohort as (
    select subject_id, hadm_id, label, cohort_group, split, age_years, is_male
    from {{ ref('stg_cohort') }}
),

labs as (
    select * from {{ ref('stg_lab_features') }}
)

select
    cohort.subject_id,
    cohort.hadm_id,
    cohort.label,
    cohort.cohort_group,
    cohort.split,
    cohort.age_years,
    cohort.is_male,
    labs.lab_itemid,
    labs.lab_name,
    labs.is_acute_marker,
    labs.aggregation_rule,
    labs.feature_value,
    labs.measurement_count,
    labs.first_measurement_hours
from cohort
inner join labs on labs.hadm_id = cohort.hadm_id
