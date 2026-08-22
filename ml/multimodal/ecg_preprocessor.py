"""
ecg_preprocessor.py
====================
Converts raw MIMIC-IV-ECG WFDB files into normalized numpy arrays ready
for 1D CNN training.

MIMIC-IV-ECG specs:
  - 12-lead ECG (I, II, III, aVR, aVL, aVF, V1–V6)
  - Sampling rate: 500 Hz
  - Duration: 10 seconds → 5000 samples per lead
  - Format: WFDB (.hea + .dat)

OUTPUT:
  data/ecg/processed/
  ├── ecg_arrays.npz   ← {hadm_id: array shape (12, 5000)}
  └── ecg_index.csv    ← hadm_id → npz key mapping

INSTALL:
  pip install wfdb
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("WARNING: wfdb not installed. Run: pip install wfdb")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ECG_MANIFEST_PATH  = "data/ecg/ecg_manifest.csv"
# wget saves under physionet.org/files/mimic-iv-ecg/1.0/ — adjust if different
ECG_WAVEFORM_DIR   = Path("data/ecg/waveforms/physionet.org/files/mimic-iv-ecg/1.0")
OUTPUT_DIR         = Path("data/ecg/processed")

TARGET_FS          = 500     # Hz — MIMIC-IV-ECG native sample rate
TARGET_LENGTH      = 5000    # samples (10 sec × 500 Hz)
N_LEADS            = 12
# ─────────────────────────────────────────────────────────────────────────────

# Standard 12-lead order
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def load_wfdb_record(record_path: str) -> np.ndarray | None:
    """
    Load a WFDB record and return array of shape (12, 5000).
    Returns None if loading fails.
    """
    if not WFDB_AVAILABLE:
        return None
    try:
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal   # shape: (samples, leads)

        # ── Resample to TARGET_LENGTH ────────────────────────────────────────
        if signal.shape[0] != TARGET_LENGTH:
            from scipy.signal import resample
            signal = resample(signal, TARGET_LENGTH, axis=0)

        # ── Reorder leads to standard order ──────────────────────────────────
        sig_names = [s.upper().strip() for s in record.sig_name]
        ordered = np.zeros((N_LEADS, TARGET_LENGTH), dtype=np.float32)
        for i, lead_name in enumerate(LEAD_ORDER):
            if lead_name in sig_names:
                idx = sig_names.index(lead_name)
                ordered[i] = signal[:, idx]
            # else: leave as zeros (lead missing)

        # ── Handle NaN/Inf in waveform ────────────────────────────────────────
        ordered = np.nan_to_num(ordered, nan=0.0, posinf=0.0, neginf=0.0)

        return ordered

    except Exception:
        return None


def normalize_ecg(ecg: np.ndarray) -> np.ndarray:
    """
    Per-lead z-score normalization.
    Each lead is independently zero-meaned and unit-variance scaled.
    Avoids cross-lead amplitude differences dominating the CNN.
    """
    mean = ecg.mean(axis=1, keepdims=True)      # (12, 1)
    std  = ecg.std(axis=1, keepdims=True) + 1e-8
    return ((ecg - mean) / std).astype(np.float32)


def resolve_record_path(row: pd.Series) -> Path | None:
    """
    Find the actual file path on disk for a given ECG record.
    Manifest paths look like: files/p1810/p18106347/s46484598/46484598
    wget saves to: data/ecg/waveforms/physionet.org/files/mimic-iv-ecg/1.0/files/...
    So we just join the manifest path onto ECG_WAVEFORM_DIR.
    """
    if pd.notna(row.get("path", None)):
        candidate = ECG_WAVEFORM_DIR / str(row["path"])
        if (Path(str(candidate) + ".hea")).exists():
            return candidate

    # Fallback: construct path from subject_id + study_id
    if pd.notna(row.get("subject_id")) and pd.notna(row.get("study_id")):
        sid  = str(int(row["subject_id"]))
        stid = str(int(row["study_id"]))
        p_folder = f"p{sid[:4]}"   # MIMIC-IV-ECG uses 4-char prefix e.g. p1188
        path = ECG_WAVEFORM_DIR / "files" / p_folder / f"p{sid}" / f"s{stid}"
        candidates = list(path.glob("*.hea"))
        if candidates:
            return candidates[0].with_suffix("")

    return None


def process_all_ecgs(manifest: pd.DataFrame) -> dict:
    """
    Process all ECG files and return dict {hadm_id: ecg_array}.
    Skips files that are missing or fail to load.
    """
    ecg_data = {}
    n_total   = len(manifest)
    n_loaded  = 0
    n_missing = 0
    n_failed  = 0

    for i, row in manifest.iterrows():
        hadm_id = int(row["hadm_id"])

        record_path = resolve_record_path(row)
        if record_path is None:
            n_missing += 1
            continue

        ecg = load_wfdb_record(str(record_path))
        if ecg is None:
            n_failed += 1
            continue

        ecg_data[hadm_id] = normalize_ecg(ecg)
        n_loaded += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_total}] loaded={n_loaded} "
                  f"missing={n_missing} failed={n_failed}")

    print(f"\nResult: {n_loaded}/{n_total} ECGs loaded "
          f"({n_missing} missing files, {n_failed} load errors)")
    return ecg_data


def save_ecg_arrays(ecg_data: dict, output_dir: Path):
    """Save ECG arrays as numpy compressed archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as npz (compressed) with hadm_id as key
    npz_path = output_dir / "ecg_arrays.npz"
    np.savez_compressed(
        str(npz_path),
        **{str(k): v for k, v in ecg_data.items()}
    )
    print(f"✓ Saved {len(ecg_data)} ECG arrays → {npz_path}")

    # Save index
    index = pd.DataFrame({
        "hadm_id": list(ecg_data.keys()),
        "has_ecg": True
    })
    index_path = output_dir / "ecg_index.csv"
    index.to_csv(index_path, index=False)
    print(f"✓ Saved ECG index → {index_path}")
    return npz_path


def main():
    print("=== ECG Preprocessor ===\n")

    if not WFDB_AVAILABLE:
        print("Install wfdb first: pip install wfdb")
        return

    # Load manifest
    manifest = pd.read_csv(ECG_MANIFEST_PATH)
    print(f"Manifest: {len(manifest)} ECG records to process")

    # Process
    ecg_data = process_all_ecgs(manifest)

    if not ecg_data:
        print("No ECGs processed. Check that waveform files are downloaded.")
        return

    # Save
    save_ecg_arrays(ecg_data, OUTPUT_DIR)

    print(f"\nECG shape per record: (12 leads, {TARGET_LENGTH} samples)")
    print("Next step: run train_multimodal.py")


if __name__ == "__main__":
    main()
