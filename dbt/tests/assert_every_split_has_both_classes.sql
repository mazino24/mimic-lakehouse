-- Stratification check: a validation split with no positives makes early
-- stopping meaningless and AUC undefined.

select
    split,
    count(*) filter (where label = 1) as positives,
    count(*) filter (where label = 0) as negatives
from {{ ref('mart_angina_training_features') }}
group by split
having count(*) filter (where label = 1) = 0
    or count(*) filter (where label = 0) = 0
