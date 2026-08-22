{{ config(materialized='view') }}

{% set source_relation = source('lake', 'model_metrics') %}

select
    run_id,
    trained_at,
    model_name,
    split,
    metric_name,
    metric_value,
    n_rows,
    n_features
from {{ source_relation }}
