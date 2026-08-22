{{ config(materialized='view') }}

-- `select *` is deliberate: the lab columns are data-dependent (which labs
-- clear the coverage threshold varies by extract), so pinning a column list
-- here would break every time the source data changes.

select * from {{ source('lake', 'feature_mart') }}
