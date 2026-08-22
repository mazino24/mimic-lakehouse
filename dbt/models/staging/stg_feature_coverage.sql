{{ config(materialized='view') }}

select
    column_name,
    column_kind,
    non_null_rows,
    total_rows,
    null_rate,
    round((1 - null_rate)::numeric, 4) as coverage_rate,
    _run_id                            as etl_run_id,
    _loaded_at                         as loaded_at
from {{ source('lake', 'feature_coverage') }}
