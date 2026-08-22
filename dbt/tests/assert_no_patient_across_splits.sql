-- The leakage test. If one patient appears in both train and test, every
-- metric downstream is optimistic and the whole study is worthless.
-- Returns offending patients; dbt fails the run if any row comes back.

select
    subject_id,
    count(distinct split) as splits_present
from {{ ref('mart_angina_training_features') }}
group by subject_id
having count(distinct split) > 1
