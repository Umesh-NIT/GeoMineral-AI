from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ML_DIR = Path("data/processed/ml")

GEO_FILE = ML_DIR / "baneya_model_features_geochemistry.csv"
STRUCT_FILE = ML_DIR / "baneya_model_features_geo_structure.csv"
TARGET_FILE = ML_DIR / "baneya_model_targets.csv"

RESULT_FILE = ML_DIR / "baneya_baseline_model_results.csv"


def evaluate_model(model, X, y, name):

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        return_train_score=False,
    )

    mae = -scores["test_mae"]
    rmse = -scores["test_rmse"]
    r2 = scores["test_r2"]

    return {
        "model": name,
        "MAE_mean": mae.mean(),
        "MAE_std": mae.std(),
        "RMSE_mean": rmse.mean(),
        "RMSE_std": rmse.std(),
        "R2_mean": r2.mean(),
        "R2_std": r2.std(),
    }


def make_models():

    return {
        "ElasticNet": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    ElasticNet(
                        alpha=0.05,
                        l1_ratio=0.5,
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        ),

        "RandomForest": Pipeline(
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
        ),

        "GradientBoosting": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=3,
                        loss="huber",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Baseline ML Models")
    print("=" * 100)

    geo = pd.read_csv(GEO_FILE)
    geo_struct = pd.read_csv(STRUCT_FILE)
    target = pd.read_csv(TARGET_FILE)

    print("\nGeochemistry dataset:")
    print(geo.shape)

    print("\nGeochemistry + Structure dataset:")
    print(geo_struct.shape)

    print("\nTarget dataset:")
    print(target.shape)

    geo = geo.merge(
        target[
            [
                "sample_id",
                "target_li_log10",
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    )

    geo_struct = geo_struct.merge(
        target[
            [
                "sample_id",
                "target_li_log10",
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    )

    X_geo = geo.drop(
        columns=[
            "sample_id",
            "target_li_log10",
        ]
    )

    X_geo_struct = geo_struct.drop(
        columns=[
            "sample_id",
            "target_li_log10",
        ]
    )

    y_geo = geo["target_li_log10"]
    y_geo_struct = geo_struct["target_li_log10"]

    results = []

    print("\n" + "=" * 100)
    print("MODEL A - GEOCHEMISTRY ONLY")
    print("=" * 100)

    for name, model in make_models().items():

        print(
            f"\nTraining {name}..."
        )

        result = evaluate_model(
            model,
            X_geo,
            y_geo,
            f"Geo_{name}",
        )

        results.append(result)

        print(
            f"MAE  : {result['MAE_mean']:.4f} "
            f"+/- {result['MAE_std']:.4f}"
        )

        print(
            f"RMSE : {result['RMSE_mean']:.4f} "
            f"+/- {result['RMSE_std']:.4f}"
        )

        print(
            f"R2   : {result['R2_mean']:.4f} "
            f"+/- {result['R2_std']:.4f}"
        )

    print("\n" + "=" * 100)
    print("MODEL B - GEOCHEMISTRY + STRUCTURE")
    print("=" * 100)

    for name, model in make_models().items():

        print(
            f"\nTraining {name}..."
        )

        result = evaluate_model(
            model,
            X_geo_struct,
            y_geo_struct,
            f"GeoStructure_{name}",
        )

        results.append(result)

        print(
            f"MAE  : {result['MAE_mean']:.4f} "
            f"+/- {result['MAE_std']:.4f}"
        )

        print(
            f"RMSE : {result['RMSE_mean']:.4f} "
            f"+/- {result['RMSE_std']:.4f}"
        )

        print(
            f"R2   : {result['R2_mean']:.4f} "
            f"+/- {result['R2_std']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "RMSE_mean"
    )

    ML_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    print("\n" + "=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()