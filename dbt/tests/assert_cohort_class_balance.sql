-- A cohort that silently collapses to one class still trains, still reports
-- 95 % accuracy, and is completely broken. Catch it at the warehouse.

with balance as (
    select
        count(*)                                          as total_rows,
        avg(label::numeric)                               as positive_share
    from {{ ref('mart_angina_training_features') }}
)

select *
from balance
where total_rows < {{ var('min_cohort_rows') }}
   or positive_share < {{ var('min_positive_share') }}
   or positive_share > {{ var('max_positive_share') }}
