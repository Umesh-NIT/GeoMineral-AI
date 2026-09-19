from pathlib import Path
import pandas as pd
import numpy as np


INPUT = Path(
    "data/processed/geochemistry/baneya_target_analysis.csv"
)

OUTPUT = Path(
    "data/processed/geochemistry/baneya_li_anomalies.csv"
)


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Lithium Anomaly Analysis")
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

    df["Li_anomaly_class"] = "background"

    df.loc[
        df["Li_value"] >= p90,
        "Li_anomaly_class"
    ] = "P90"

    df.loc[
        df["Li_value"] >= p95,
        "Li_anomaly_class"
    ] = "P95"

    df.loc[
        df["Li_value"] >= p97_5,
        "Li_anomaly_class"
    ] = "P97.5"

    print("\nThresholds:")
    print(f"P90   = {p90:.4f}")
    print(f"P95   = {p95:.4f}")
    print(f"P97.5 = {p97_5:.4f}")

    print("\n" + "=" * 100)
    print("ANOMALY COUNTS")
    print("=" * 100)

    print(
        df["Li_anomaly_class"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 100)
    print("P95+ LITHIUM SAMPLES")
    print("=" * 100)

    columns = [
        "sample_id",
        "sample_category",
        "latitude",
        "longitude",
        "sample_type",
        "Li_value",
        "Li_anomaly_class",
    ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    p95_samples = (
        df[
            df["Li_value"] >= p95
        ]
        .sort_values(
            "Li_value",
            ascending=False
        )
    )

    print(
        p95_samples[
            columns
        ].to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("P90+ SAMPLES BY SAMPLE CATEGORY")
    print("=" * 100)

    p90_samples = df[
        df["Li_value"] >= p90
    ]

    print(
        p90_samples[
            "sample_category"
        ]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 100)
    print("P95+ SAMPLES BY SAMPLE TYPE")
    print("=" * 100)

    print(
        p95_samples[
            "sample_type"
        ]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 100)
    print("P95+ SPATIAL EXTENT")
    print("=" * 100)

    if len(p95_samples) > 0:

        print(
            "Latitude:",
            p95_samples["latitude"].min(),
            "to",
            p95_samples["latitude"].max()
        )

        print(
            "Longitude:",
            p95_samples["longitude"].min(),
            "to",
            p95_samples["longitude"].max()
        )

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()