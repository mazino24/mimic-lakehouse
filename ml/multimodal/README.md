# Multimodal consumers (tabular + 12-lead ECG)

These four modules are the research side of the project: a 1D-CNN over
MIMIC-IV-ECG waveforms fused with the tabular lab features, plus the
preprocessing needed to get from PhysioNet WFDB files to model-ready arrays.

They are **consumers of the platform**, not part of it. The lakehouse is what
feeds them:

| Module | What it needs | Where the platform provides it |
| --- | --- | --- |
| `filter_ecg.py` | which ECG studies to download | `silver.ecg_records` — already matched to admissions in the pipeline |
| `ecg_preprocessor.py` | downloaded WFDB files | `data/ecg/waveforms/` after the PhysioNet fetch |
| `train_multimodal.py` | a flat tabular feature table | `scripts/export_training_csv.py` |
| `multimodal_model.py` | — | model definition only |

## Running them against the lakehouse

```bash
# 1. Export the mart in the shape these scripts expect
python scripts/export_training_csv.py --out data/processed/tabular_features.csv

# 2. Waveforms (real MIMIC-IV-ECG only; needs PhysioNet credentials)
python ml/multimodal/ecg_preprocessor.py

# 3. Train the fusion model
python ml/multimodal/train_multimodal.py
```

## Why the ECG matching moved into Spark

`record_list.csv` indexes ~800k ECG studies and carries no `hadm_id`. Matching
a study to the stay whose window contains it is a join against every
admission — a pandas merge that used to run out of memory. It now lives in
`spark/lakehouse/transforms/cleaning.py::clean_ecg_records`, is covered by a
unit test, and produces one ECG per stay (the earliest, i.e. the one taken at
presentation).

The waveform bytes themselves stay out of the warehouse. The mart carries the
`ecg_path` pointer; the arrays live in object storage. Putting 2.8 GB of
float32 into Postgres would be the wrong call.
