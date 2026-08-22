-- Acute markers must use the first in-window draw, everything else the mean.
-- This is the pipeline's central clinical rule; assert it survived the trip
-- through Spark, Parquet and JDBC.

select
    lab_name,
    is_acute_marker,
    aggregation_rule,
    count(*) as offending_rows
from {{ ref('stg_lab_features') }}
where (is_acute_marker and aggregation_rule <> 'first')
   or (not is_acute_marker and aggregation_rule <> 'mean')
group by lab_name, is_acute_marker, aggregation_rule
