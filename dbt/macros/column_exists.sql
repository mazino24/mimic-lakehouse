{#
  The feature mart's lab columns are data-dependent: which labs survive the
  coverage filter depends on what was actually measured in the source extract.
  Models that reference a specific lab guard with this macro so a thin
  synthetic dataset does not break `dbt run`.
#}
{% macro column_exists(relation, column_name) %}
    {%- if execute -%}
        {%- set columns = adapter.get_columns_in_relation(relation) | map(attribute="name")
              | map("lower") | list -%}
        {{ return(column_name | lower in columns) }}
    {%- else -%}
        {{ return(false) }}
    {%- endif -%}
{% endmacro %}


{% macro column_or_null(relation, column_name) %}
    {%- if column_exists(relation, column_name) -%}
        {{ adapter.quote(column_name) }}
    {%- else -%}
        cast(null as numeric)
    {%- endif -%}
{% endmacro %}
