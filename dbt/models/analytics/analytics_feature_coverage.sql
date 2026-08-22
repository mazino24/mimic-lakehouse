-- How complete each feature is, and whether it is complete enough to model on.

select
    column_name,
    column_kind,
    total_rows,
    non_null_rows,
    null_rate,
    coverage_rate,
    case
        when null_rate <= 0.10 then 'complete'
        when null_rate <= 0.50 then 'partial'
        when null_rate <= 0.90 then 'sparse'
        else 'mostly_missing'
    end                as completeness_tier,
    etl_run_id,
    loaded_at
from {{ ref('stg_feature_coverage') }}
order by null_rate
