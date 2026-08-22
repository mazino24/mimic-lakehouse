{#
  Every numeric lab column on the feature mart, excluding keys, metadata and
  demographics. Used to build the coverage/profile analytics models without
  hard-coding a lab list that changes with the source extract.
#}
{% macro lab_feature_columns(relation) %}
    {%- set excluded = [
        "subject_id", "hadm_id", "label", "split", "split_bucket", "gender",
        "gender_male", "age", "anchor_age", "admittime", "dischtime", "los_hours",
        "admit_year", "admission_type", "insurance", "race", "died_in_hospital",
        "has_ecg", "ecg_study_id", "ecg_path", "ecg_hours_from_admit",
        "feature_count", "_run_id", "_loaded_at", "_source"
    ] -%}
    {%- if execute -%}
        {%- set columns = [] -%}
        {%- for column in adapter.get_columns_in_relation(relation) -%}
            {%- if column.name | lower not in excluded and column.is_number() -%}
                {%- do columns.append(column.name) -%}
            {%- endif -%}
        {%- endfor -%}
        {{ return(columns) }}
    {%- else -%}
        {{ return([]) }}
    {%- endif -%}
{% endmacro %}
