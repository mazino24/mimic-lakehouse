{#
  Use the custom schema name verbatim instead of dbt's default
  `<target_schema>_<custom_schema>` concatenation.

  Without this, a model configured with `+schema: marts` lands in
  `analytics_marts`, while `warehouse/init/01_schemas.sql`, the runbook, the
  training job and every documented query all refer to `marts`. Layer names
  are part of the warehouse's public interface here, so they are pinned.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
