from lakehouse.transforms.naming import dedupe_columns, normalize_column


def test_normalizes_real_mimic_lab_labels():
    assert normalize_column("Cholesterol, LDL, Calculated") == "cholesterol_ldl_calculated"
    assert normalize_column("Troponin T") == "troponin_t"
    assert normalize_column("  Creatinine, Whole Blood ") == "creatinine_whole_blood"


def test_symbols_become_words_not_holes():
    assert normalize_column("% Hemoglobin A1c") == "pct_hemoglobin_a1c"
    assert normalize_column("Protein/Creatinine Ratio") == "protein_per_creatinine_ratio"


def test_identifier_never_starts_with_a_digit():
    # Postgres rejects an unquoted identifier starting with a digit.
    assert normalize_column("24 hr Creatinine").startswith("lab_24")


def test_truncation_keeps_columns_unique():
    long_a = "Alpha " * 20 + "one"
    long_b = "Alpha " * 20 + "two"
    a, b = dedupe_columns([normalize_column(long_a), normalize_column(long_b)])
    assert a != b
    assert len(a) <= 60 and len(b) <= 60


def test_empty_label_does_not_produce_empty_identifier():
    assert normalize_column("///") == "per_per_per"[:55] or normalize_column("///")
    assert normalize_column("") == "unnamed"
