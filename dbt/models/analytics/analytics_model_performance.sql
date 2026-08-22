-- Model metrics next to the data-quality history that explains them.

with metrics as (
    select * from {{ ref('stg_model_metrics') }}
),

wide as (
    select
        run_id,
        trained_at,
        model_name,
        split,
        max(n_rows)                                        as n_rows,
        max(n_features)                                    as n_features,
        max(metric_value) filter (where metric_name = 'roc_auc')   as roc_auc,
        max(metric_value) filter (where metric_name = 'pr_auc')    as pr_auc,
        max(metric_value) filter (where metric_name = 'f1')        as f1,
        max(metric_value) filter (where metric_name = 'recall')    as recall,
        max(metric_value) filter (where metric_name = 'precision') as precision,
        max(metric_value) filter (where metric_name = 'accuracy')  as accuracy
    from metrics
    group by run_id, trained_at, model_name, split
)

select
    *,
    roc_auc - lag(roc_auc) over (
        partition by model_name, split order by trained_at
    ) as roc_auc_change_vs_previous_run
from wide
order by trained_at desc, split, roc_auc desc nulls last
