from pathlib import Path
import pandas as pd
import numpy as np


INPUT = Path(
    "data/processed/geochemistry/baneya_geochemical_standardized.csv"
)

OUTPUT = Path(
    "data/processed/geochemistry/baneya_target_analysis.csv"
)


def percentile_rank(series):
    return series.rank(pct=True)


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Lithium / REE Target Analysis")
    print("=" * 100)

    df = pd.read_csv(INPUT)

    li = pd.to_numeric(
        df["Li_value"],
        errors="coerce"
    )

    df["Li_log10"] = np.log10(
        li.clip(lower=0.000001)
    )

    df["Li_percentile"] = percentile_rank(li)

    thresholds = [
        50,
        75,
        90,
        95,
        97.5,
        99
    ]

    print("\n" + "=" * 100)
    print("LITHIUM DISTRIBUTION")
    print("=" * 100)

    print(
        li.describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.975,
                0.99
            ]
        ).to_string()
    )

    print("\n" + "=" * 100)
    print("LITHIUM PERCENTILE THRESHOLDS")
    print("=" * 100)

    for p in thresholds:

        value = np.percentile(
            li.dropna(),
            p
        )

        print(
            f"P{p:>5}: {value:.4f}"
        )

    print("\n" + "=" * 100)
    print("SAMPLES ABOVE LITHIUM THRESHOLDS")
    print("=" * 100)

    for p in thresholds:

        threshold = np.percentile(
            li.dropna(),
            p
        )

        count = (li >= threshold).sum()

        print(
            f"P{p:>5} ({threshold:.4f}): {count} samples"
        )

    print("\n" + "=" * 100)
    print("TOP 20 LITHIUM SAMPLES")
    print("=" * 100)

    columns = [
        "sample_id",
        "sample_category",
        "latitude",
        "longitude",
        "sample_type",
        "Li",
        "Li_value",
        "Li_censored",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    top = (
        df[
            available
        ]
        .sort_values(
            "Li_value",
            ascending=False
        )
        .head(20)
    )

    print(
        top.to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("LITHIUM BY SAMPLE CATEGORY")
    print("=" * 100)

    category_stats = (
        df.groupby("sample_category")["Li_value"]
        .agg(
            count="count",
            min="min",
            median="median",
            mean="mean",
            max="max",
        )
        .sort_values(
            "median",
            ascending=False
        )
    )

    print(
        category_stats.to_string()
    )

    ree_columns = [
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

    available_ree = [
        column
        for column in ree_columns
        if column in df.columns
    ]

    print("\n" + "=" * 100)
    print("REE AVAILABILITY")
    print("=" * 100)

    for column in available_ree:

        print(
            f"{column:15} "
            f"non-null={df[column].notna().sum():3d} "
            f"missing={df[column].isna().sum():3d}"
        )

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()