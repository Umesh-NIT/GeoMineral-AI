from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "data/processed/geochemistry/baneya_target_structural.csv"
)

OUTPUT_DIR = Path(
    "data/processed/ml"
)

GEO_OUTPUT = OUTPUT_DIR / "baneya_model_features_geochemistry.csv"
STRUCT_OUTPUT = OUTPUT_DIR / "baneya_model_features_geo_structure.csv"
TARGET_OUTPUT = OUTPUT_DIR / "baneya_model_targets.csv"


def numeric(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Reduced Model Features")
    print("=" * 100)

    df = pd.read_csv(INPUT)

    print("\nInput shape:", df.shape)

    # ------------------------------------------------------------------
    # Basic identifiers
    # ------------------------------------------------------------------

    base = pd.DataFrame({
        "sample_id": df["sample_id"],
        "latitude": numeric(df, "latitude"),
        "longitude": numeric(df, "longitude"),
    })

    # ------------------------------------------------------------------
    # Selected individual geochemical predictors
    # ------------------------------------------------------------------

    geo = pd.DataFrame({
        "Cs": numeric(df, "Cs*_value"),
        "Th": numeric(df, "Th_value"),
        "Rb": numeric(df, "Rb_value"),
        "U": numeric(df, "U_value"),
        "Sn": numeric(df, "Sn_value"),
        "Be": numeric(df, "Be_value"),
        "Nb": numeric(df, "Nb*_value"),
        "Ta": numeric(df, "Ta_value"),
    })

    # ------------------------------------------------------------------
    # REE features
    # ------------------------------------------------------------------

    ree_names = [
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ]

    ree = pd.DataFrame()

    for name in ree_names:

        column = f"{name}_value"

        if column in df.columns:

            ree[name] = numeric(
                df,
                column
            )

    # Total REE

    geo["REE_sum"] = ree.sum(
        axis=1,
        min_count=1
    )

    # Light REE

    lree_names = [
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Sm",
    ]

    geo["LREE_sum"] = ree[
        [
            x for x in lree_names
            if x in ree.columns
        ]
    ].sum(
        axis=1,
        min_count=1
    )

    # Heavy REE

    hree_names = [
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ]

    geo["HREE_sum"] = ree[
        [
            x for x in hree_names
            if x in ree.columns
        ]
    ].sum(
        axis=1,
        min_count=1
    )

    # LREE / HREE ratio

    geo["LREE_HREE_ratio"] = (
        geo["LREE_sum"]
        /
        geo["HREE_sum"].replace(
            0,
            np.nan
        )
    )

    # Eu anomaly.
    #
    # Simple local anomaly using Sm and Gd:
    #
    # Eu_anomaly = Eu / sqrt(Sm * Gd)

    sm = numeric(
        df,
        "Sm_value"
    )

    eu = numeric(
        df,
        "Eu_value"
    )

    gd = numeric(
        df,
        "Gd_value"
    )

    denominator = np.sqrt(
        sm.clip(lower=0)
        *
        gd.clip(lower=0)
    )

    geo["Eu_anomaly"] = (
        eu / denominator.replace(
            0,
            np.nan
        )
    )

    # ------------------------------------------------------------------
    # Log transforms
    # ------------------------------------------------------------------

    log_features = [
        "Cs",
        "Th",
        "Rb",
        "U",
        "Sn",
        "Be",
        "Nb",
        "Ta",
        "REE_sum",
        "LREE_sum",
        "HREE_sum",
    ]

    geo_log = pd.DataFrame(
        index=df.index
    )

    for column in log_features:

        geo_log[
            f"{column}_log10"
        ] = np.log10(
            geo[column].clip(
                lower=1e-6
            )
        )

    # ------------------------------------------------------------------
    # Structural predictors
    # ------------------------------------------------------------------

    structure = pd.DataFrame({
        "nearest_structure_distance_m":
            numeric(
                df,
                "nearest_structure_distance_m"
            ),

        "nearest_structure_strike":
            numeric(
                df,
                "nearest_structure_strike"
            ),

        "nearest_structure_dip":
            numeric(
                df,
                "nearest_structure_dip"
            ),

        "nearest_structure_dip_azimuth":
            numeric(
                df,
                "nearest_structure_dip_azimuth"
            ),

        "structures_within_250m":
            numeric(
                df,
                "structures_within_250m"
            ),

        "structures_within_500m":
            numeric(
                df,
                "structures_within_500m"
            ),

        "structures_within_1000m":
            numeric(
                df,
                "structures_within_1000m"
            ),
    })

    # ------------------------------------------------------------------
    # Model A: Geochemistry only
    # ------------------------------------------------------------------

    geo_features = pd.concat(
        [
            base,
            geo,
            geo_log,
        ],
        axis=1
    )

    # ------------------------------------------------------------------
    # Model B: Geochemistry + Structure
    # ------------------------------------------------------------------

    geo_structure_features = pd.concat(
        [
            geo_features,
            structure,
        ],
        axis=1
    )

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    targets = df[
        [
            "sample_id",
            "target_li_ppm",
            "target_li_log10",
            "li_anomaly_p90",
            "li_anomaly_p95",
            "li_anomaly_p97_5",
            "li_anomaly_level",
        ]
    ].copy()

    # ------------------------------------------------------------------
    # Remove completely empty columns
    # ------------------------------------------------------------------

    geo_features = geo_features.dropna(
        axis=1,
        how="all"
    )

    geo_structure_features = (
        geo_structure_features.dropna(
            axis=1,
            how="all"
        )
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    geo_features.to_csv(
        GEO_OUTPUT,
        index=False
    )

    geo_structure_features.to_csv(
        STRUCT_OUTPUT,
        index=False
    )

    targets.to_csv(
        TARGET_OUTPUT,
        index=False
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("MODEL A - GEOCHEMISTRY ONLY")
    print("=" * 100)

    print(
        "Shape:",
        geo_features.shape
    )

    print(
        "Features:",
        geo_features.shape[1] - 1
    )

    print("\n" + "=" * 100)
    print("MODEL B - GEOCHEMISTRY + STRUCTURE")
    print("=" * 100)

    print(
        "Shape:",
        geo_structure_features.shape
    )

    print(
        "Features:",
        geo_structure_features.shape[1] - 1
    )

    print("\n" + "=" * 100)
    print("TARGET")
    print("=" * 100)

    print(
        "Shape:",
        targets.shape
    )

    print("\nTarget distribution:")

    print(
        targets[
            "li_anomaly_level"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMissing values - Model A:")

    missing_a = (
        geo_features
        .drop(columns=["sample_id"])
        .isna()
        .sum()
    )

    print(
        missing_a[
            missing_a > 0
        ].to_string()
        if (missing_a > 0).any()
        else "None"
    )

    print("\nMissing values - Model B:")

    missing_b = (
        geo_structure_features
        .drop(columns=["sample_id"])
        .isna()
        .sum()
    )

    print(
        missing_b[
            missing_b > 0
        ].to_string()
        if (missing_b > 0).any()
        else "None"
    )

    print("\nSaved:")

    print(GEO_OUTPUT)
    print(STRUCT_OUTPUT)
    print(TARGET_OUTPUT)


if __name__ == "__main__":
    main()