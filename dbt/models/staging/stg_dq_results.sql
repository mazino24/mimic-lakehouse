{{ config(materialized='view') }}

select
    run_id,
    checked_at,
    layer,
    table_name,
    check_name,
    severity,
    (passed = 'true')                            as passed,
    observed,
    threshold,
    row_count,
    details,
    split_part(check_name, '__', 1)              as check_type
from {{ source('lake', 'dq_results') }}
