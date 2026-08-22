-- Case-vs-control lab profile: which markers actually separate the groups.
-- This is the model-independent evidence that the cohort definition is sane;
-- if troponin does not separate here, no classifier will save you.

with panel as (
    select * from {{ ref('int_cohort_lab_panel') }}
),

by_group as (
    select
        lab_name,
        is_acute_marker,
        aggregation_rule,
        cohort_group,
        count(*)                                                     as n_admissions,
        avg(feature_value)                                           as mean_value,
        stddev_samp(feature_value)                                   as sd_value,
        percentile_cont(0.5) within group (order by feature_value)   as median_value
    from panel
    group by lab_name, is_acute_marker, aggregation_rule, cohort_group
),

pivoted as (
    select
        lab_name,
        is_acute_marker,
        aggregation_rule,
        max(n_admissions)  filter (where cohort_group = 'angina')  as angina_admissions,
        max(n_admissions)  filter (where cohort_group = 'control') as control_admissions,
        max(mean_value)    filter (where cohort_group = 'angina')  as angina_mean,
        max(mean_value)    filter (where cohort_group = 'control') as control_mean,
        max(median_value)  filter (where cohort_group = 'angina')  as angina_median,
        max(median_value)  filter (where cohort_group = 'control') as control_median,
        max(sd_value)      filter (where cohort_group = 'angina')  as angina_sd,
        max(sd_value)      filter (where cohort_group = 'control') as control_sd
    from by_group
    group by lab_name, is_acute_marker, aggregation_rule
),

scored as (
    select
        *,
        angina_mean - control_mean as mean_difference,
        -- Cohen's d: effect size, so a 0.01 mg/dL difference on a tight
        -- distribution is not mistaken for a weak signal on a wide one.
        case
            when coalesce(angina_sd, 0) + coalesce(control_sd, 0) = 0 then null
            else round(
                ((angina_mean - control_mean)
                 / sqrt(((coalesce(angina_sd, 0) ^ 2) + (coalesce(control_sd, 0) ^ 2)) / 2.0)
                )::numeric, 3)
        end as cohens_d
    from pivoted
)

-- Postgres only allows a bare alias in ORDER BY, never an alias inside an
-- expression, so the ranking happens one level up from where cohens_d is
-- computed.
select *
from scored
order by abs(coalesce(cohens_d, 0)) desc
