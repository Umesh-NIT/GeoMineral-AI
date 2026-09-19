from pathlib import Path

import pandas as pd
import numpy as np


INPUT = Path(
    "data/processed/geochemistry/baneya_target_structural.csv"
)

OUTPUT_DIR = Path(
    "data/processed/ml"
)

FEATURE_OUTPUT = OUTPUT_DIR / "baneya_features.csv"
TARGET_OUTPUT = OUTPUT_DIR / "baneya_target.csv"
DICTIONARY_OUTPUT = OUTPUT_DIR / "baneya_feature_dictionary.csv"


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


GEOCHEMICAL_FEATURES = [
    "Be_value",
    "Cs*_value",
    "Nb*_value",
    "Rb_value",
    "Sn_value",
    "Ta_value",
    "Th_value",
    "U_value",
    "Y*_value",
    "La_value",
    "Ce_value",
    "Pr_value",
    "Nd_value",
    "Sm_value",
    "Eu_value",
    "Gd_value",
    "Tb_value",
    "Dy_value",
    "Ho_value",
    "Er_value",
    "Tm_value",
    "Yb_value",
    "Lu_value",
]


STRUCTURAL_FEATURES = [
    "nearest_structure_distance_m",
    "nearest_structure_strike",
    "nearest_structure_dip",
    "nearest_structure_dip_azimuth",
    "structures_within_250m",
    "structures_within_500m",
    "structures_within_1000m",
    "structure_density_250m",
    "structure_density_500m",
    "structure_density_1000m",
]


LOCATION_FEATURES = [
    "latitude",
    "longitude",
]


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya ML Feature Dataset")
    print("=" * 100)

    df = pd.read_csv(INPUT)

    print("\nInput shape:")
    print(df.shape)

    selected_features = []

    for column in LOCATION_FEATURES:
        if column in df.columns:
            selected_features.append(column)

    for column in GEOCHEMICAL_FEATURES:
        if column in df.columns:
            selected_features.append(column)

    for column in STRUCTURAL_FEATURES:
        if column in df.columns:
            selected_features.append(column)

    selected_features = list(
        dict.fromkeys(selected_features)
    )

    missing_features = [
        column
        for column in (
            LOCATION_FEATURES
            + GEOCHEMICAL_FEATURES
            + STRUCTURAL_FEATURES
        )
        if column not in df.columns
    ]

    print("\nSelected predictor features:")
    for column in selected_features:
        print(" ", column)

    if missing_features:

        print("\nWARNING - Missing expected features:")

        for column in missing_features:
            print(" ", column)

    print(
        "\nTotal predictor features:",
        len(selected_features)
    )

    features = df[
        selected_features
    ].copy()

    target = pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "latitude": df["latitude"],
            "longitude": df["longitude"],
            "target_li_ppm": df["target_li_ppm"],
            "target_li_log10": df["target_li_log10"],
            "li_anomaly_p90": df["li_anomaly_p90"],
            "li_anomaly_p95": df["li_anomaly_p95"],
            "li_anomaly_p97_5": df["li_anomaly_p97_5"],
            "li_anomaly_level": df["li_anomaly_level"],
        }
    )

    features.insert(
        0,
        "sample_id",
        df["sample_id"]
    )

    print("\nFeature missingness:")

    missing_summary = (
        features
        .drop(columns=["sample_id"])
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing_summary[
            missing_summary > 0
        ].to_string()
        if (missing_summary > 0).any()
        else "No missing predictor values."
    )

    dictionary_rows = []

    for column in selected_features:

        if column in LOCATION_FEATURES:
            category = "location"

        elif column in GEOCHEMICAL_FEATURES:
            category = "geochemistry"

        elif column in STRUCTURAL_FEATURES:
            category = "structure"

        else:
            category = "other"

        numeric = pd.to_numeric(
            features[column],
            errors="coerce"
        )

        dictionary_rows.append(
            {
                "feature": column,
                "category": category,
                "dtype": str(
                    features[column].dtype
                ),
                "non_null": int(
                    numeric.notna().sum()
                ),
                "missing": int(
                    numeric.isna().sum()
                ),
                "missing_pct": float(
                    numeric.isna().mean() * 100
                ),
                "unique": int(
                    numeric.nunique()
                ),
                "min": numeric.min(),
                "median": numeric.median(),
                "max": numeric.max(),
            }
        )

    dictionary = pd.DataFrame(
        dictionary_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        FEATURE_OUTPUT,
        index=False
    )

    target.to_csv(
        TARGET_OUTPUT,
        index=False
    )

    dictionary.to_csv(
        DICTIONARY_OUTPUT,
        index=False
    )

    print("\n" + "=" * 100)
    print("FINAL DATASET")
    print("=" * 100)

    print(
        "\nFeatures shape:",
        features.shape
    )

    print(
        "Target shape:",
        target.shape
    )

    print(
        "Feature dictionary shape:",
        dictionary.shape
    )

    print("\nFeature categories:")

    print(
        dictionary[
            "category"
        ].value_counts().to_string()
    )

    print("\nSaved:")

    print(
        FEATURE_OUTPUT
    )

    print(
        TARGET_OUTPUT
    )

    print(
        DICTIONARY_OUTPUT
    )

    print("\nTarget distribution:")

    print(
        target[
            "li_anomaly_level"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()