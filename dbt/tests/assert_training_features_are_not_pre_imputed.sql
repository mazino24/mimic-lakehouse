-- Guards against a well-meaning "fix" that fills NULLs in the mart.
--
-- In real clinical data every lab has missingness: a test that was never
-- ordered has no value. A *lab* column with a zero null rate therefore means
-- somebody imputed upstream and quietly leaked test-set statistics into
-- training — the exact bug this pipeline was rebuilt to eliminate.
--
-- Demographics and stay attributes are legitimately complete, so only
-- `lab_feature` columns are considered.

select
    column_name,
    column_kind,
    null_rate
from {{ ref('analytics_feature_coverage') }}
where column_kind = 'lab_feature'
  and null_rate = 0
