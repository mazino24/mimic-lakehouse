"""Column-name normalisation.

MIMIC lab labels are free text: ``"Cholesterol, LDL, Calculated"``,
``"% Hemoglobin A1c"``. Those become physical column names after the pivot, so
they have to survive Parquet, Postgres (63-char identifier limit) and dbt.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")
_MULTI_UNDERSCORE = re.compile(r"_+")

_REPLACEMENTS = {
    "%": "pct",
    "/": " per ",
    "+": " plus ",
    "#": " num ",
}


def normalize_column(name: str, *, max_length: int = 55) -> str:
    """``"Cholesterol, LDL, Calculated"`` -> ``"cholesterol_ldl_calculated"``."""
    text = name.strip().lower()
    for symbol, word in _REPLACEMENTS.items():
        text = text.replace(symbol, f" {word} ")
    text = _NON_ALNUM.sub("_", text)
    text = _MULTI_UNDERSCORE.sub("_", text).strip("_")
    if not text:
        text = "unnamed"
    if text[0].isdigit():
        text = f"lab_{text}"
    return text[:max_length].rstrip("_")


def dedupe_columns(names: list[str]) -> list[str]:
    """Guarantee uniqueness after truncation (``foo``, ``foo_2``, ``foo_3``)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out
