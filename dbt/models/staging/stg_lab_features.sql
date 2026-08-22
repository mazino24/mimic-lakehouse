{{ config(materialized='view') }}

with source as (
    select * from {{ source('lake', 'lab_features') }}
)

select
    hadm_id,
    itemid                    as lab_itemid,
    label                     as lab_name,
    is_acute_marker,
    aggregation               as aggregation_rule,
    feature_value,
    first_value,
    mean_value,
    min_value,
    max_value,
    measurement_count,
    first_measurement_hours,
    _run_id                   as etl_run_id
from source
