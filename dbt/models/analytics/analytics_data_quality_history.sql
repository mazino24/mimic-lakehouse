-- Data quality as a time series: pass rate per table per run, plus the last
-- failure for anything currently unhealthy.

with results as (
    select * from {{ ref('stg_dq_results') }}
),

per_run as (
    select
        run_id,
        layer,
        table_name,
        max(checked_at)                                     as checked_at,
        count(*)                                            as checks_run,
        count(*) filter (where passed)                      as checks_passed,
        count(*) filter (where not passed and severity = 'error') as blocking_failures,
        count(*) filter (where not passed and severity = 'warn')  as warnings,
        max(row_count)                                      as row_count
    from results
    group by run_id, layer, table_name
)

select
    *,
    round((checks_passed::numeric / nullif(checks_run, 0)), 4) as pass_rate,
    (blocking_failures = 0)                                    as run_healthy
from per_run
order by checked_at desc, layer, table_name
