-- Executive summary of the study population: the table you put on slide 2.

with cohort as (
    select * from {{ ref('stg_cohort') }}
)

select
    cohort_group,
    split,
    count(*)                                                    as patients,
    round(avg(age_years)::numeric, 1)                           as mean_age,
    percentile_cont(0.5) within group (order by age_years)      as median_age,
    round(avg(is_male)::numeric, 3)                             as male_share,
    round(avg(length_of_stay_hours)::numeric, 1)                as mean_los_hours,
    round(avg(case when died_in_hospital then 1 else 0 end)::numeric, 4) as in_hospital_mortality,
    min(admitted_at)                                            as earliest_admission,
    max(admitted_at)                                            as latest_admission
from cohort
group by cohort_group, split
order by cohort_group, split
