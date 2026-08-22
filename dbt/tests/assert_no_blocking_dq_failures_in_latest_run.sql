-- The warehouse should never contain data from a run whose blocking checks
-- failed. If this fires, someone published around the gate.

with latest_run as (
    select run_id
    from {{ ref('stg_dq_results') }}
    order by checked_at desc
    limit 1
)

select
    results.layer,
    results.table_name,
    results.check_name,
    results.details
from {{ ref('stg_dq_results') }} as results
inner join latest_run on latest_run.run_id = results.run_id
where results.severity = 'error'
  and not results.passed
