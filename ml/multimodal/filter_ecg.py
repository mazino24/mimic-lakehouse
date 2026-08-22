"""
filter_ecg.py
=============
Filters MIMIC-IV-ECG records so you only download what you need.

MIMIC-IV-ECG structure on PhysioNet:
  mimic-iv-ecg/
  ├── record_list.csv          ← index of all 800k+ ECG studies
  └── files/
      └── p{subject_id[:2]}/p{subject_id}/s{study_id}/
          ├── *.hea            ← header (metadata)
          └── *.dat            ← waveform binary

WORKFLOW:
  Step 1  → Run this script with record_list.csv to get your download list
  Step 2  → Use wget/rsync with that list to download only matched files
  Step 3  → Run ecg_preprocessor.py to convert waveforms to numpy arrays
"""

from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these paths
# ─────────────────────────────────────────────────────────────────────────────
COHORT_PATH      = "data/processed/final_cohort.csv"
RECORD_LIST_PATH = "data/raw/mimic-iv-ecg/record_list.csv"   # from PhysioNet
ADMISSIONS_PATH  = "data/raw/admissions.csv"
OUTPUT_DIR       = Path("data/ecg")
# ─────────────────────────────────────────────────────────────────────────────


def load_record_list(path: str) -> pd.DataFrame:
    """Load MIMIC-IV-ECG record list. Columns vary by version."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    # Standardize column names across MIMIC-IV-ECG versions
    rename = {}
    for col in df.columns:
        if "subject" in col:
            rename[col] = "subject_id"
        elif "study" in col:
            rename[col] = "study_id"
        elif "path" in col or "filename" in col or "record" in col:
            rename[col] = "path"
        elif "ecg_time" in col or "chartdate" in col or "start" in col:
            rename[col] = "ecg_time"
    df = df.rename(columns=rename)
    return df


def match_ecg_to_admissions(
    cohort: pd.DataFrame,
    admissions: pd.DataFrame,
    records: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match ECG studies to admissions by:
      1. Same subject_id
      2. ECG recorded during the admission window (admittime → dischtime)

    Returns DataFrame with columns:
      hadm_id, subject_id, label, ecg_time, study_id, path
    """
    if "ecg_time" in records.columns:
        records["ecg_time"] = pd.to_datetime(records["ecg_time"])

    # final_cohort.csv already contains admittime/dischtime — use them directly.
    # If missing (older cohort file), fall back to admissions.csv merge.
    if "admittime" not in cohort.columns or "dischtime" not in cohort.columns:
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])
        admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
        cohort = cohort.merge(
            admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
            on=["subject_id", "hadm_id"],
            how="left"
        )

    cohort = cohort.copy()
    cohort["admittime"] = pd.to_datetime(cohort["admittime"])
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"])

    # Merge on subject_id to get candidate ECG records
    matched = cohort.merge(records, on="subject_id", how="inner")

    # Filter to ECG within admission window (if time info available)
    if "ecg_time" in matched.columns:
        in_window = (
            (matched["ecg_time"] >= matched["admittime"]) &
            (matched["ecg_time"] <= matched["dischtime"])
        )
        matched = matched[in_window]
        print(f"ECGs within admission window: {len(matched)}")
    else:
        print("WARNING: No ecg_time column — cannot filter by admission window.")
        print("         All ECGs for matched subjects will be included.")

    # For each admission, keep only the FIRST ECG (closest to admission)
    if "ecg_time" in matched.columns:
        matched = (
            matched
            .sort_values("ecg_time")
            .drop_duplicates(subset=["hadm_id"], keep="first")
        )

    print(f"Unique admissions with ECG: {matched['hadm_id'].nunique()}")
    print(f"  Positive (angina):  {matched[matched['label']==1]['hadm_id'].nunique()}")
    print(f"  Negative (control): {matched[matched['label']==0]['hadm_id'].nunique()}")

    return matched


def build_download_manifest(matched: pd.DataFrame, output_dir: Path) -> Path:
    """
    Writes two files:
      - ecg_manifest.csv    → mapping hadm_id → study_id → path (for your code)
      - ecg_download.txt    → list of paths for wget/rsync
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest for your ML code
    cols = ["hadm_id", "subject_id", "label", "study_id", "path"]
    cols = [c for c in cols if c in matched.columns]
    manifest = matched[cols].reset_index(drop=True)
    manifest_path = output_dir / "ecg_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\n✓ Manifest saved: {manifest_path}  ({len(manifest)} records)")

    # Save download list
    if "path" in matched.columns:
        download_path = output_dir / "ecg_download.txt"
        with open(download_path, "w") as f:
            for p in matched["path"].dropna().unique():
                # Write both .hea and .dat
                base = str(p).replace(".hea", "").replace(".dat", "")
                f.write(base + ".hea\n")
                f.write(base + ".dat\n")
        print(f"✓ Download list: {download_path}  ({len(matched)} studies)")
        print(
            f"\n── HOW TO DOWNLOAD ──────────────────────────────────────────\n"
            f"Replace YOUR_USERNAME and YOUR_PASSWORD below:\n\n"
            f"  wget -r -N -c -np --user=YOUR_USERNAME --password=YOUR_PASSWORD \\\n"
            f"    -i {download_path} \\\n"
            f"    -P data/ecg/waveforms/ \\\n"
            f"    https://physionet.org/files/mimic-iv-ecg/1.0/\n\n"
            f"Or with rsync (faster):\n"
            f"  rsync -CazvP --no-relative \\\n"
            f"    rsync.physionet.org::mimic-iv-ecg/1.0/ data/ecg/waveforms/ \\\n"
            f"    --files-from={download_path}\n"
            f"──────────────────────────────────────────────────────────────"
        )
        return download_path
    return manifest_path


def main():
    print("=== MIMIC-IV-ECG Filter ===\n")

    # 1. Load cohort
    cohort = pd.read_csv(COHORT_PATH)
    print(f"Cohort: {len(cohort)} admissions "
          f"({cohort['label'].value_counts().to_dict()})")

    # 2. Load admissions for time windows
    admissions = pd.read_csv(ADMISSIONS_PATH)

    # 3. Load ECG record list
    if not Path(RECORD_LIST_PATH).exists():
        print(f"\nERROR: record_list.csv not found at {RECORD_LIST_PATH}")
        print("Download it first (it's tiny, ~50MB for the full index):")
        print("  wget --user=USER --password=PASS \\")
        print("    https://physionet.org/files/mimic-iv-ecg/1.0/record_list.csv \\")
        print("    -P data/raw/mimic-iv-ecg/")
        return

    records = load_record_list(RECORD_LIST_PATH)
    print(f"ECG record list: {len(records)} studies, "
          f"{records['subject_id'].nunique()} subjects")

    # 4. Match
    matched = match_ecg_to_admissions(cohort, admissions, records)

    if matched.empty:
        print("\nWARNING: No ECG records matched the cohort!")
        print("Check that subject_id values align between datasets.")
        return

    # 5. Build download manifest
    build_download_manifest(matched, OUTPUT_DIR)

    # 6. Coverage report
    total = len(cohort)
    covered = matched["hadm_id"].nunique()
    print("\n── Coverage ─────────────────────────────────────────────────")
    print(f"  Admissions with ECG: {covered}/{total} ({covered/total:.1%})")
    print(f"  Admissions without ECG: {total - covered}")
    print("  → Missing ECG will be handled as None in multimodal model")
    print("────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
