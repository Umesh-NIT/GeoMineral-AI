from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


STRUCTURE_FILE = Path(
    "data/interim/ngdr/baneya/tables_extracted/Tables/Oriented_Structure_Plane_LSM.xls"
)

TARGET_FILE = Path(
    "data/processed/geochemistry/baneya_target_dataset.csv"
)

OUTPUT = Path(
    "data/processed/geochemistry/baneya_target_structural.csv"
)


EARTH_RADIUS_M = 6371008.8


def haversine_distance_matrix(
    target_lat,
    target_lon,
    structure_lat,
    structure_lon,
):
    target_rad = np.radians(
        np.column_stack(
            [target_lat, target_lon]
        )
    )

    structure_rad = np.radians(
        np.column_stack(
            [structure_lat, structure_lon]
        )
    )

    tree = BallTree(
        structure_rad,
        metric="haversine"
    )

    return tree, target_rad


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Structural Feature Engineering")
    print("=" * 100)

    structures = pd.read_excel(
        STRUCTURE_FILE,
        sheet_name="All_data"
    )

    targets = pd.read_csv(
        TARGET_FILE
    )

    print("\nStructural observations:", len(structures))
    print("Geochemical target points:", len(targets))

    structures["Latitude"] = pd.to_numeric(
        structures["Latitude"],
        errors="coerce"
    )

    structures["Longitude"] = pd.to_numeric(
        structures["Longitude"],
        errors="coerce"
    )

    structures["STRIKE"] = pd.to_numeric(
        structures["STRIKE"],
        errors="coerce"
    )

    structures["DIP"] = pd.to_numeric(
        structures["DIP"],
        errors="coerce"
    )

    structures["DIP_AZIMUTH"] = pd.to_numeric(
        structures["DIP_AZIMUTH"],
        errors="coerce"
    )

    targets["latitude"] = pd.to_numeric(
        targets["latitude"],
        errors="coerce"
    )

    targets["longitude"] = pd.to_numeric(
        targets["longitude"],
        errors="coerce"
    )

    structures = structures.dropna(
        subset=["Latitude", "Longitude"]
    ).reset_index(drop=True)

    targets = targets.dropna(
        subset=["latitude", "longitude"]
    ).reset_index(drop=True)

    print(
        "\nStructures with coordinates:",
        len(structures)
    )

    print(
        "Targets with coordinates:",
        len(targets)
    )

    structure_coords = np.radians(
        structures[
            ["Latitude", "Longitude"]
        ].to_numpy()
    )

    target_coords = np.radians(
        targets[
            ["latitude", "longitude"]
        ].to_numpy()
    )

    tree = BallTree(
        structure_coords,
        metric="haversine"
    )

    print("\nCalculating nearest structural observations...")

    distances, indices = tree.query(
        target_coords,
        k=1
    )

    nearest_distance_m = (
        distances[:, 0] * EARTH_RADIUS_M
    )

    nearest_index = indices[:, 0]

    nearest = structures.iloc[
        nearest_index
    ].reset_index(drop=True)

    targets[
        "nearest_structure_distance_m"
    ] = nearest_distance_m

    targets[
        "nearest_structure_point_type"
    ] = nearest["POINT_TYPE"].to_numpy()

    targets[
        "nearest_structure_strike"
    ] = nearest["STRIKE"].to_numpy()

    targets[
        "nearest_structure_dip"
    ] = nearest["DIP"].to_numpy()

    targets[
        "nearest_structure_dip_azimuth"
    ] = nearest["DIP_AZIMUTH"].to_numpy()

    print(
        "\nNearest structure distance statistics:"
    )

    print(
        targets[
            "nearest_structure_distance_m"
        ].describe().to_string()
    )

    print(
        "\nCalculating structural density..."
    )

    target_coords_rad = target_coords

    for radius_m in [250, 500, 1000]:

        radius_rad = (
            radius_m / EARTH_RADIUS_M
        )

        neighbours = tree.query_radius(
            target_coords_rad,
            r=radius_rad,
            count_only=False
        )

        counts = np.array(
            [
                len(x)
                for x in neighbours
            ]
        )

        targets[
            f"structures_within_{radius_m}m"
        ] = counts

        area_km2 = np.pi * (
            radius_m / 1000
        ) ** 2

        targets[
            f"structure_density_{radius_m}m"
        ] = counts / area_km2

    print("\nStructural density summary:")

    density_columns = [
        "structures_within_250m",
        "structures_within_500m",
        "structures_within_1000m",
        "structure_density_250m",
        "structure_density_500m",
        "structure_density_1000m",
    ]

    print(
        targets[
            density_columns
        ].describe().to_string()
    )

    print("\nNearest structure types:")

    print(
        targets[
            "nearest_structure_point_type"
        ]
        .value_counts(dropna=False)
        .to_string()
    )

    targets.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)

    print("\nFinal shape:")
    print(targets.shape)


if __name__ == "__main__":
    main()