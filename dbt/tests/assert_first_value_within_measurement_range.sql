-- Sanity check on the aggregation itself: the promoted value must lie inside
-- the observed min/max for that admission and lab.

select
    hadm_id,
    lab_name,
    feature_value,
    min_value,
    max_value
from {{ ref('stg_lab_features') }}
where feature_value is not null
  and (feature_value < min_value - 1e-9 or feature_value > max_value + 1e-9)
