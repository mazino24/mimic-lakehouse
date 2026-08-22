#!/usr/bin/env python3
"""Generate a synthetic, MIMIC-IV-shaped dataset.

MIMIC-IV is credentialed data: it cannot be committed to a public repo, and a
reviewer without a PhysioNet account cannot run the pipeline at all. This
script emits CSVs with the *same schemas, join keys, quirks and failure modes*
as the real extracts, so ``make demo`` exercises every stage end to end.

Deliberately reproduced quirks
------------------------------
* ICD codes stored without dots (``I25.110`` -> ``I25110``)
* lab draws recorded outside the admission window (the window filter must bite)
* text-only lab results where ``valuenum`` is NULL
* serial troponin draws per stay (so first-vs-mean aggregation matters)
* duplicate diagnosis rows and a few impossible discharge timestamps
* ~90 % of the wide lab panel missing, as in the real data

The clinical signal is synthetic but directionally real: cases carry higher
troponin / CK-MB and worse lipid panels, so a model trained on the demo data
lands in a believable AUC range instead of 0.5 or 1.0.

    python scripts/generate_synthetic_mimic.py --patients 5000 --out-dir data/raw
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# (itemid, label, fluid, category, is_acute, healthy_mean, healthy_sd, case_shift, prevalence)
LAB_PANEL = [
    (51003, "Troponin T", "Blood", "Chemistry", True, 0.04, 0.055, 0.10, 0.55),
    (50908, "CK-MB Index", "Blood", "Chemistry", True, 2.1, 2.4, 2.2, 0.30),
    (50852, "% Hemoglobin A1c", "Blood", "Chemistry", False, 5.5, 1.1, 0.6, 0.18),
    (50904, "Cholesterol, HDL", "Blood", "Chemistry", False, 55.0, 16.0, -6.0, 0.22),
    (50905, "Cholesterol, LDL, Measured", "Blood", "Chemistry", False, 105.0, 34.0, 14.0, 0.16),
    (50907, "Cholesterol, Total", "Blood", "Chemistry", False, 185.0, 42.0, 12.0, 0.24),
    (50931, "Glucose", "Blood", "Chemistry", False, 105.0, 32.0, 10.0, 0.92),
    (50912, "Creatinine", "Blood", "Chemistry", False, 1.0, 0.32, 0.24, 0.95),
    (51222, "Hemoglobin", "Blood", "Hematology", False, 13.4, 1.7, -0.9, 0.94),
    (51221, "Hematocrit", "Blood", "Hematology", False, 40.0, 5.0, -2.4, 0.93),
    (51265, "Platelet Count", "Blood", "Hematology", False, 245.0, 65.0, -12.0, 0.90),
    (50971, "Potassium", "Blood", "Chemistry", False, 4.1, 0.45, 0.1, 0.88),
    (50983, "Sodium", "Blood", "Chemistry", False, 139.0, 3.2, -0.6, 0.88),
    (50882, "Bicarbonate", "Blood", "Chemistry", False, 24.5, 3.0, -0.8, 0.80),
    (51006, "Urea Nitrogen", "Blood", "Chemistry", False, 16.0, 6.5, 3.2, 0.85),
    (50862, "Albumin", "Blood", "Chemistry", False, 4.0, 0.5, -0.3, 0.35),
    (51301, "White Blood Cells", "Blood", "Hematology", False, 8.2, 2.6, 1.4, 0.90),
    (50960, "Magnesium", "Blood", "Chemistry", False, 2.0, 0.25, 0.0, 0.45),
    # Rare labs, mostly null — the coverage filter should drop these.
    (51100, "Creatinine, Ascites", "Ascites", "Chemistry", False, 1.1, 0.4, 0.0, 0.004),
    (51101, "Glucose, Joint Fluid", "Joint Fluid", "Chemistry", False, 95.0, 20.0, 0.0, 0.003),
]

ANGINA_CODES = ["I200", "I201", "I208", "I209", "I25110", "I25118", "I2510", "I25119"]
OTHER_CARDIAC_CODES = ["I5021", "I2109", "I4891", "I639", "I110"]
NON_CARDIAC_CODES = [
    "E119", "J449", "K219", "N179", "M545", "F329", "R079", "Z9861",
    "J189", "K802", "E785", "G4733", "L03115", "S72001A", "R1084",
]
ADMISSION_TYPES = ["EW EMER.", "URGENT", "ELECTIVE", "OBSERVATION ADMIT", "DIRECT EMER."]
INSURANCE = ["Medicare", "Medicaid", "Private", "Other"]
RACE = ["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO", "ASIAN", "OTHER", "UNKNOWN"]
MARITAL = ["MARRIED", "SINGLE", "WIDOWED", "DIVORCED", ""]


def write_csv(path: Path, header: list[str], rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def fmt(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def generate(out_dir: Path, n_patients: int, seed: int, ecg_share: float) -> dict[str, int]:
    rng = random.Random(seed)
    stats: dict[str, int] = {}

    patients, admissions, diagnoses, labevents, chartevents, ecg_records = [], [], [], [], [], []
    subject_id, hadm_id, labevent_id, study_id = 10_000_000, 20_000_000, 1, 40_000_000

    for _ in range(n_patients):
        subject_id += 1
        is_case = rng.random() < 0.45
        # A slice of the non-cases carry other cardiac disease: they must be
        # excluded from the control group, not silently used as negatives.
        is_other_cardiac = (not is_case) and rng.random() < 0.18

        # Angina is heterogeneous: some cases present with markedly abnormal
        # markers, many present with an almost normal panel. A single fixed
        # shift per lab would make the classes trivially separable and the
        # demo model would score an implausible AUC.
        if is_case:
            expressivity = rng.betavariate(1.3, 2.2) * 1.8
        elif rng.random() < 0.12:
            # A minority of controls carry subclinical abnormalities, so the
            # classes overlap in both directions rather than separating
            # cleanly on a single marker.
            expressivity = rng.betavariate(1.2, 3.0) * 0.9
        else:
            expressivity = 0.0

        gender = rng.choice(["M", "F"])
        base_age = rng.gauss(65 if is_case else 58, 15)
        age = int(min(91, max(18, base_age)))
        anchor_year = rng.randint(2110, 2200)
        died = rng.random() < 0.04
        dod = ""
        if died:
            dod = fmt(datetime(
                anchor_year + rng.randint(0, 5), rng.randint(1, 12), rng.randint(1, 28)
            ))
        patients.append([subject_id, gender, age, anchor_year, "2014 - 2016", dod])

        for _ in range(rng.choices([1, 2, 3], weights=[70, 22, 8])[0]):
            hadm_id += 1
            admit = datetime(anchor_year, rng.randint(1, 12), rng.randint(1, 28),
                             rng.randint(0, 23), rng.randint(0, 59))
            los_hours = max(4, int(rng.gauss(96 if is_case else 72, 48)))
            discharge = admit + timedelta(hours=los_hours)
            # 0.3 % corrupt records: discharge before admission. The silver
            # layer is expected to drop these.
            if rng.random() < 0.003:
                discharge = admit - timedelta(hours=5)
            expire_flag = 1 if (died and rng.random() < 0.1) else 0

            admissions.append([
                subject_id, hadm_id, fmt(admit), fmt(discharge),
                fmt(discharge) if expire_flag else "", rng.choice(ADMISSION_TYPES),
                f"P{rng.randint(1000, 9999)}XY", "EMERGENCY ROOM", "HOME",
                rng.choice(INSURANCE), "ENGLISH", rng.choice(MARITAL), rng.choice(RACE),
                fmt(admit - timedelta(hours=3)), fmt(admit - timedelta(minutes=20)), expire_flag,
            ])

            seq = 1
            codes: list[str] = []
            if is_case:
                codes.append(rng.choice(ANGINA_CODES))
            if is_other_cardiac:
                codes.append(rng.choice(OTHER_CARDIAC_CODES))
            codes += rng.sample(NON_CARDIAC_CODES, rng.randint(1, 5))
            for code in codes:
                diagnoses.append([subject_id, hadm_id, seq, code, 10])
                seq += 1
                # Duplicate diagnosis rows exist in the real table.
                if rng.random() < 0.02:
                    diagnoses.append([subject_id, hadm_id, seq, code, 10])
                    seq += 1
            if rng.random() < 0.15:  # legacy ICD-9 rows, must be ignored
                diagnoses.append([subject_id, hadm_id, seq, str(rng.randint(4100, 4999)), 9])

            for itemid, _label, _fluid, _cat, is_acute, mean, sd, shift, prevalence in LAB_PANEL:
                if rng.random() > prevalence:
                    continue
                draws = rng.randint(2, 4) if is_acute else rng.randint(1, 3)
                for draw in range(draws):
                    value = rng.gauss(mean + shift * expressivity, sd)
                    if is_acute and draw > 0:
                        # Serial draws trend back toward baseline after
                        # treatment — this is why "first" beats "mean".
                        value = max(0.0, value * rng.uniform(0.55, 0.9))
                    value = round(max(0.0, value), 3)
                    charttime = admit + timedelta(
                        hours=rng.uniform(0.2, max(1.0, los_hours * 0.9))
                    )
                    # 4 % of draws are outpatient labs outside the stay window.
                    if rng.random() < 0.04:
                        charttime = admit - timedelta(days=rng.randint(20, 200))
                    text_only = rng.random() < 0.02
                    labevents.append([
                        labevent_id, subject_id, hadm_id, rng.randint(10**7, 10**8 - 1),
                        itemid, f"P{rng.randint(1000, 9999)}AB", fmt(charttime),
                        fmt(charttime + timedelta(minutes=rng.randint(20, 180))),
                        "PENDING" if text_only else str(value),
                        "" if text_only else value, "mg/dL",
                        "", "", "abnormal" if rng.random() < 0.25 else "",
                        rng.choice(["ROUTINE", "STAT"]), "",
                    ])
                    labevent_id += 1

            if rng.random() < 0.6:
                chartevents.append([
                    subject_id, hadm_id, rng.randint(3 * 10**7, 4 * 10**7),
                    rng.randint(1000, 99999),
                    fmt(admit + timedelta(hours=1)), fmt(admit + timedelta(hours=2)), 220179,
                    str(rng.randint(90, 180)), rng.randint(90, 180), "mmHg", 0,
                ])

            if rng.random() < ecg_share:
                study_id += 1
                ecg_time = admit + timedelta(hours=rng.uniform(0.1, 24))
                ecg_records.append([
                    subject_id, study_id, study_id, fmt(ecg_time),
                    f"files/p{str(subject_id)[:4]}/p{subject_id}/s{study_id}/{study_id}",
                ])

    stats["patients"] = write_csv(
        out_dir / "patients.csv",
        ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
        patients,
    )
    stats["admissions"] = write_csv(
        out_dir / "admissions.csv",
        ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type",
         "admit_provider_id", "admission_location", "discharge_location", "insurance",
         "language", "marital_status", "race", "edregtime", "edouttime", "hospital_expire_flag"],
        admissions,
    )
    stats["diagnoses_icd"] = write_csv(
        out_dir / "diagnoses_icd.csv",
        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        diagnoses,
    )
    stats["d_labitems"] = write_csv(
        out_dir / "d_labitems.csv",
        ["itemid", "label", "fluid", "category"],
        [[i, label, fluid, category] for i, label, fluid, category, *_ in LAB_PANEL],
    )
    stats["labevents"] = write_csv(
        out_dir / "labevents.csv",
        ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "order_provider_id",
         "charttime", "storetime", "value", "valuenum", "valueuom", "ref_range_lower",
         "ref_range_upper", "flag", "priority", "comments"],
        labevents,
    )
    stats["chartevents"] = write_csv(
        out_dir / "chartevents.csv",
        ["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid",
         "value", "valuenum", "valueuom", "warning"],
        chartevents,
    )
    stats["ecg_record_list"] = write_csv(
        out_dir / "mimic-iv-ecg" / "record_list.csv",
        ["subject_id", "study_id", "file_name", "ecg_time", "path"],
        ecg_records,
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ecg-share", type=float, default=0.35,
                        help="Share of admissions that get an ECG study")
    args = parser.parse_args()

    stats = generate(args.out_dir, args.patients, args.seed, args.ecg_share)
    width = max(len(name) for name in stats)
    print(f"synthetic MIMIC-IV extract written to {args.out_dir.resolve()}")
    for name, count in stats.items():
        print(f"  {name:<{width}}  {count:>9,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
