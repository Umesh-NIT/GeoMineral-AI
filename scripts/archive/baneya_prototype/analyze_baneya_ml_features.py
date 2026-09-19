from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_FILE = Path(
    "data/processed/ml/baneya_features.csv"
)

TARGET_FILE = Path(
    "data/processed/ml/baneya_target.csv"
)

OUTPUT_DIR = Path(
    "data/processed/ml"
)

CORRELATION_OUTPUT = (
    OUTPUT_DIR / "baneya_feature_correlations.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "baneya_ml_feature_summary.csv"
)


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya ML Feature Analysis")
    print("=" * 100)

    X = pd.read_csv(FEATURE_FILE)
    y = pd.read_csv(TARGET_FILE)

    print("\nFeatures:")
    print(X.shape)

    print("\nTargets:")
    print(y.shape)

    data = X.merge(
        y[
            [
                "sample_id",
                "target_li_ppm",
                "target_li_log10",
                "li_anomaly_level",
            ]
        ],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    numeric_features = [
        c
        for c in X.columns
        if c != "sample_id"
        and pd.api.types.is_numeric_dtype(X[c])
    ]

    print(
        "\nNumeric predictors:",
        len(numeric_features)
    )

    records = []

    target = pd.to_numeric(
        data["target_li_log10"],
        errors="coerce"
    )

    for feature in numeric_features:

        values = pd.to_numeric(
            data[feature],
            errors="coerce"
        )

        valid = (
            values.notna()
            & target.notna()
        )

        if valid.sum() >= 3:

            pearson = values[valid].corr(
                target[valid],
                method="pearson"
            )

            spearman = values[valid].corr(
                target[valid],
                method="spearman"
            )

        else:

            pearson = np.nan
            spearman = np.nan

        records.append(
            {
                "feature": feature,
                "non_null": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "unique": int(values.nunique()),
                "pearson_li_log10": pearson,
                "spearman_li_log10": spearman,
                "abs_spearman": (
                    abs(spearman)
                    if pd.notna(spearman)
                    else np.nan
                ),
            }
        )

    correlation = pd.DataFrame(records)

    correlation = correlation.sort_values(
        "abs_spearman",
        ascending=False
    )

    print("\n" + "=" * 100)
    print("FEATURE RELATIONSHIP WITH LITHIUM")
    print("=" * 100)

    print(
        correlation[
            [
                "feature",
                "non_null",
                "missing",
                "pearson_li_log10",
                "spearman_li_log10",
            ]
        ]
        .head(35)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("TOP FEATURES BY |SPEARMAN|")
    print("=" * 100)

    print(
        correlation[
            [
                "feature",
                "spearman_li_log10",
                "abs_spearman",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("HIGH CORRELATION BETWEEN PREDICTORS")
    print("=" * 100)

    feature_matrix = data[
        numeric_features
    ].copy()

    corr_matrix = feature_matrix.corr(
        method="spearman"
    )

    pairs = []

    for i in range(len(numeric_features)):

        for j in range(i + 1, len(numeric_features)):

            a = numeric_features[i]
            b = numeric_features[j]

            value = corr_matrix.loc[a, b]

            if pd.notna(value) and abs(value) >= 0.90:

                pairs.append(
                    {
                        "feature_1": a,
                        "feature_2": b,
                        "spearman_correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )

    redundant = pd.DataFrame(pairs)

    if len(redundant) > 0:

        redundant = redundant.sort_values(
            "absolute_correlation",
            ascending=False
        )

        print(
            redundant.to_string(
                index=False
            )
        )

    else:

        print(
            "No predictor pairs with |Spearman| >= 0.90."
        )

    print("\n" + "=" * 100)
    print("ANOMALY GROUP SUMMARY")
    print("=" * 100)

    anomaly_summary = (
        data.groupby(
            "li_anomaly_level"
        )["target_li_ppm"]
        .agg(
            [
                "count",
                "min",
                "median",
                "mean",
                "max",
            ]
        )
    )

    print(
        anomaly_summary.to_string()
    )

    summary = correlation.copy()

    summary["feature_role"] = "candidate"

    summary.loc[
        summary["abs_spearman"] >= 0.50,
        "feature_role"
    ] = "strong_candidate"

    summary.loc[
        summary["abs_spearman"] < 0.20,
        "feature_role"
    ] = "weak_candidate"

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    correlation.to_csv(
        CORRELATION_OUTPUT,
        index=False
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(CORRELATION_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()