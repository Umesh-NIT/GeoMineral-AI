from pathlib import Path
import pandas as pd
import numpy as np


INPUT = Path(
    "data/processed/geochemistry/baneya_li_anomalies.csv"
)

OUTPUT = Path(
    "data/processed/geochemistry/baneya_target_dataset.csv"
)


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Target Dataset")
    print("=" * 100)

    df = pd.read_csv(INPUT)

    df["Li_value"] = pd.to_numeric(
        df["Li_value"],
        errors="coerce"
    )

    df["Li_log10"] = np.log10(
        df["Li_value"].clip(lower=0.000001)
    )

    p90 = df["Li_value"].quantile(0.90)
    p95 = df["Li_value"].quantile(0.95)
    p97_5 = df["Li_value"].quantile(0.975)

    # Continuous target
    df["target_li_ppm"] = df["Li_value"]

    df["target_li_log10"] = df["Li_log10"]

    # Statistical anomaly indicators
    df["li_anomaly_p90"] = (
        df["Li_value"] >= p90
    ).astype(int)

    df["li_anomaly_p95"] = (
        df["Li_value"] >= p95
    ).astype(int)

    df["li_anomaly_p97_5"] = (
        df["Li_value"] >= p97_5
    ).astype(int)

    # Ordered anomaly class
    df["li_anomaly_level"] = 0

    df.loc[
        df["Li_value"] >= p90,
        "li_anomaly_level"
    ] = 1

    df.loc[
        df["Li_value"] >= p95,
        "li_anomaly_level"
    ] = 2

    df.loc[
        df["Li_value"] >= p97_5,
        "li_anomaly_level"
    ] = 3

    print("\nThresholds:")
    print(f"P90   = {p90:.4f}")
    print(f"P95   = {p95:.4f}")
    print(f"P97.5 = {p97_5:.4f}")

    print("\nTarget counts:")

    print(
        "P90:",
        df["li_anomaly_p90"].sum()
    )

    print(
        "P95:",
        df["li_anomaly_p95"].sum()
    )

    print(
        "P97.5:",
        df["li_anomaly_p97_5"].sum()
    )

    print("\nAnomaly levels:")

    print(
        df["li_anomaly_level"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nHighest Li samples:")

    columns = [
        "sample_id",
        "sample_category",
        "latitude",
        "longitude",
        "sample_type",
        "Li_value",
        "target_li_log10",
        "li_anomaly_level",
    ]

    print(
        df[
            columns
        ]
        .sort_values(
            "Li_value",
            ascending=False
        )
        .head(15)
        .to_string(index=False)
    )

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)

    print("\nFinal shape:")
    print(df.shape)


if __name__ == "__main__":
    main()