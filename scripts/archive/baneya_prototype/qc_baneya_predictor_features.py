from pathlib import Path

import pandas as pd
import numpy as np


INPUT = Path(
    "data/processed/geochemistry/baneya_target_structural.csv"
)

OUTPUT = Path(
    "data/processed/geochemistry/baneya_predictor_qc.csv"
)


TARGET_COLUMNS = {
    "Li",
    "Li_value",
    "Li_censored",
    "Li_log10",
    "Li_percentile",
    "Li_anomaly_class",
    "target_li_ppm",
    "target_li_log10",
    "li_anomaly_p90",
    "li_anomaly_p95",
    "li_anomaly_p97_5",
    "li_anomaly_level",
}


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Predictor Feature QC")
    print("=" * 100)

    df = pd.read_csv(INPUT)

    print("\nDataset shape:", df.shape)

    predictor_columns = []

    for column in df.columns:

        if column in TARGET_COLUMNS:
            continue

        if column.endswith("_censored"):
            continue

        if column.endswith("_log10"):
            continue

        if column in {
            "sample_id",
            "sample_category",
            "sample_type",
            "sample_media",
            "commodity",
            "area_block",
            "remarks",
            "toposheet_no",
            "nearest_structure_point_type",
        }:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if numeric.notna().sum() >= 10:

            predictor_columns.append(column)

    print("\nCandidate numeric predictors:", len(predictor_columns))

    records = []

    for column in predictor_columns:

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        records.append(
            {
                "feature": column,
                "non_null": int(numeric.notna().sum()),
                "missing": int(numeric.isna().sum()),
                "missing_pct": float(
                    numeric.isna().mean() * 100
                ),
                "unique": int(
                    numeric.nunique()
                ),
                "min": numeric.min(),
                "median": numeric.median(),
                "mean": numeric.mean(),
                "max": numeric.max(),
                "std": numeric.std(),
            }
        )

    qc = pd.DataFrame(records)

    qc = qc.sort_values(
        ["missing_pct", "feature"]
    )

    print("\n" + "=" * 100)
    print("NUMERIC PREDICTOR QUALITY")
    print("=" * 100)

    print(
        qc.to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("HIGH-MISSING FEATURES (>50%)")
    print("=" * 100)

    high_missing = qc[
        qc["missing_pct"] > 50
    ]

    if high_missing.empty:
        print("None")
    else:
        print(
            high_missing[
                [
                    "feature",
                    "missing",
                    "missing_pct",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 100)
    print("CONSTANT / NEAR-CONSTANT FEATURES")
    print("=" * 100)

    near_constant = qc[
        qc["unique"] <= 2
    ]

    if near_constant.empty:
        print("None")
    else:
        print(
            near_constant[
                [
                    "feature",
                    "unique",
                    "min",
                    "median",
                    "max",
                ]
            ].to_string(index=False)
        )

    qc.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()