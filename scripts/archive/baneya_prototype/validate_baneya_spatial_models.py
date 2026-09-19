from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ML_DIR = Path("data/processed/ml")

GEO_FILE = (
    ML_DIR /
    "baneya_model_features_geochemistry.csv"
)

STRUCT_FILE = (
    ML_DIR /
    "baneya_model_features_geo_structure.csv"
)

TARGET_FILE = (
    ML_DIR /
    "baneya_model_targets.csv"
)

OUTPUT = (
    ML_DIR /
    "baneya_spatial_validation_results.csv"
)


def create_spatial_blocks(df, n_lat=3, n_lon=3):

    df = df.copy()

    lat_bins = np.linspace(
        df["latitude"].min(),
        df["latitude"].max(),
        n_lat + 1
    )

    lon_bins = np.linspace(
        df["longitude"].min(),
        df["longitude"].max(),
        n_lon + 1
    )

    df["lat_block"] = pd.cut(
        df["latitude"],
        bins=lat_bins,
        labels=False,
        include_lowest=True
    )

    df["lon_block"] = pd.cut(
        df["longitude"],
        bins=lon_bins,
        labels=False,
        include_lowest=True
    )

    df["spatial_block"] = (
        df["lat_block"].astype(str)
        + "_"
        + df["lon_block"].astype(str)
    )

    return df


def evaluate_spatial(
    features,
    target,
    model_name,
):

    data = features.merge(
        target[
            [
                "sample_id",
                "target_li_log10",
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    )

    data = create_spatial_blocks(data)

    feature_columns = [
        c
        for c in features.columns
        if c not in {
            "sample_id",
            "latitude",
            "longitude",
        }
    ]

    results = []

    blocks = sorted(
        data["spatial_block"].dropna().unique()
    )

    print(
        f"\n{model_name}: "
        f"{len(blocks)} spatial blocks"
    )

    for block in blocks:

        train = data[
            data["spatial_block"] != block
        ]

        test = data[
            data["spatial_block"] == block
        ]

        if len(test) < 5:

            continue

        X_train = train[
            feature_columns
        ]

        y_train = train[
            "target_li_log10"
        ]

        X_test = test[
            feature_columns
        ]

        y_test = test[
            "target_li_log10"
        ]

        model = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            prediction
        )

        rmse = mean_squared_error(
                            y_test,
                            prediction,
                    
                        )

        r2 = r2_score(
            y_test,
            prediction
        ) if len(test) >= 2 else np.nan

        results.append(
            {
                "model": model_name,
                "spatial_block": block,
                "train_samples": len(train),
                "test_samples": len(test),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(
            f"Block {block}: "
            f"train={len(train)}, "
            f"test={len(test)}, "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R2={r2:.4f}"
        )

    return results


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Spatial Model Validation")
    print("=" * 100)

    geo = pd.read_csv(
        GEO_FILE
    )

    geo_struct = pd.read_csv(
        STRUCT_FILE
    )

    target = pd.read_csv(
        TARGET_FILE
    )

    print("\nGeo features:", geo.shape)
    print(
        "Geo + Structure features:",
        geo_struct.shape
    )
    print(
        "Targets:",
        target.shape
    )

    results = []

    results.extend(
        evaluate_spatial(
            geo,
            target,
            "Geo_RandomForest"
        )
    )

    results.extend(
        evaluate_spatial(
            geo_struct,
            target,
            "GeoStructure_RandomForest"
        )
    )

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 100)
    print("SPATIAL VALIDATION RESULTS")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("AVERAGE SPATIAL PERFORMANCE")
    print("=" * 100)

    summary = (
        results_df
        .groupby("model")
        [
            [
                "MAE",
                "RMSE",
                "R2",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    print(
        summary.to_string()
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
