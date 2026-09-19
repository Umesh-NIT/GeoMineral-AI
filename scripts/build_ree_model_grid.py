from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "occurrences"
    / "ree_occurrence_points.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial"
)

POSITIVE_FILE = OUTPUT_DIR / "ree_positive_samples.csv"
BACKGROUND_FILE = OUTPUT_DIR / "ree_background_samples.csv"
GRID_FILE = OUTPUT_DIR / "ree_model_grid.csv"


BUFFER_KM = 25
GRID_SIZE_KM = 1
BACKGROUND_RATIO = 1


def main():

    print("=" * 80)
    print("GeoMineral AI - REE Model Grid Builder")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "project_id",
            "latitude",
            "longitude",
        ]
    ).copy()

    positive = (
        df[
            [
                "project_id",
                "exp_upid",
                "project_title",
                "state",
                "district",
                "exploration_stage",
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates(
            subset=[
                "latitude",
                "longitude",
            ]
        )
        .copy()
    )

    positive["target"] = 1

    print(f"\nUnique positive locations: {len(positive)}")

    lat_mean = positive["latitude"].mean()

    km_per_degree_lat = 111.32

    km_per_degree_lon = (
        111.32 * np.cos(
            np.radians(lat_mean)
        )
    )

    lat_buffer = BUFFER_KM / km_per_degree_lat

    lon_buffer = BUFFER_KM / km_per_degree_lon

    min_lat = positive["latitude"].min() - lat_buffer
    max_lat = positive["latitude"].max() + lat_buffer

    min_lon = positive["longitude"].min() - lon_buffer
    max_lon = positive["longitude"].max() + lon_buffer

    lat_step = GRID_SIZE_KM / km_per_degree_lat

    lon_step = GRID_SIZE_KM / km_per_degree_lon

    print("\nModeling region:")
    print(
        f"Latitude: {min_lat:.6f} -> {max_lat:.6f}"
    )
    print(
        f"Longitude: {min_lon:.6f} -> {max_lon:.6f}"
    )

    print("\nGenerating 1 km grid...")

    lat_values = np.arange(
        min_lat,
        max_lat + lat_step,
        lat_step,
    )

    lon_values = np.arange(
        min_lon,
        max_lon + lon_step,
        lon_step,
    )

    print(
        f"Grid rows: {len(lat_values):,}"
    )

    print(
        f"Grid columns: {len(lon_values):,}"
    )

    total_cells = (
        len(lat_values)
        * len(lon_values)
    )

    print(
        f"Potential grid cells: "
        f"{total_cells:,}"
    )

    if total_cells > 5_000_000:

        raise RuntimeError(
            "Grid is too large. "
            "Reduce modeling extent or resolution."
        )

    lat_grid, lon_grid = np.meshgrid(
        lat_values,
        lon_values,
        indexing="ij",
    )

    grid = pd.DataFrame(
        {
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel(),
        }
    )

    print(
        f"\nGenerated grid cells: "
        f"{len(grid):,}"
    )

    # Approximate distance in kilometers.
    # Used only for selecting the 25 km modeling region.
    lat_scale = 111.32

    lon_scale = (
        111.32
        * np.cos(
            np.radians(
                grid["latitude"]
            )
        )
    )

    # Keep cells within BUFFER_KM of at least
    # one known REE occurrence.
    occurrence_lat = positive[
        "latitude"
    ].to_numpy()

    occurrence_lon = positive[
        "longitude"
    ].to_numpy()

    min_distance = np.full(
        len(grid),
        np.inf,
    )

    chunk_size = 50_000

    print(
        "\nSelecting cells near known REE "
        "occurrences..."
    )

    for start in range(
        0,
        len(grid),
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            len(grid),
        )

        chunk = grid.iloc[
            start:end
        ]

        lat_diff = (
            chunk["latitude"].to_numpy()[
                :, None
            ]
            - occurrence_lat[None, :]
        ) * lat_scale

        lon_scale_chunk = (
            111.32
            * np.cos(
                np.radians(
                    chunk["latitude"]
                    .to_numpy()
                )
            )
        )

        lon_diff = (
            chunk["longitude"].to_numpy()[
                :, None
            ]
            - occurrence_lon[None, :]
        ) * lon_scale_chunk[:, None]

        distance = np.sqrt(
            lat_diff ** 2
            + lon_diff ** 2
        )

        min_distance[start:end] = (
            distance.min(axis=1)
        )

    grid["distance_to_known_ree_km"] = (
        min_distance
    )

    modeling_grid = grid[
        grid[
            "distance_to_known_ree_km"
        ]
        <= BUFFER_KM
    ].copy()

    modeling_grid["target"] = 0

    # Mark cells containing known REE locations.
    # A small tolerance is used because the grid
    # centers may not exactly equal occurrence coordinates.
    tolerance_km = GRID_SIZE_KM / 2

    positive_mask = (
        modeling_grid[
            "distance_to_known_ree_km"
        ]
        <= tolerance_km
    )

    modeling_grid.loc[
        positive_mask,
        "target"
    ] = 1

    positive_grid = modeling_grid[
        modeling_grid["target"] == 1
    ].copy()

    background_grid = modeling_grid[
        modeling_grid["target"] == 0
    ].copy()

    print(
        f"\nPositive grid cells: "
        f"{len(positive_grid):,}"
    )

    print(
        f"Available background cells: "
        f"{len(background_grid):,}"
    )

    desired_background = (
        len(positive_grid)
        * BACKGROUND_RATIO
    )

    if len(background_grid) > desired_background:

        background = (
            background_grid
            .sample(
                n=desired_background,
                random_state=42,
            )
            .copy()
        )

    else:

        background = background_grid.copy()

    background["target"] = 0

    positive_output = positive_grid[
        [
            "latitude",
            "longitude",
            "distance_to_known_ree_km",
            "target",
        ]
    ].copy()

    background_output = background[
        [
            "latitude",
            "longitude",
            "distance_to_known_ree_km",
            "target",
        ]
    ].copy()

    final_grid = pd.concat(
        [
            positive_output,
            background_output,
        ],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive_output.to_csv(
        POSITIVE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    background_output.to_csv(
        BACKGROUND_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_grid.to_csv(
        GRID_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("MODEL GRID SUMMARY")
    print("=" * 80)

    print(
        f"\nPositive samples: "
        f"{len(positive_output):,}"
    )

    print(
        f"Background samples: "
        f"{len(background_output):,}"
    )

    print(
        f"Total training grid samples: "
        f"{len(final_grid):,}"
    )

    print("\nTarget distribution:")

    print(
        final_grid["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSaved:")

    print(POSITIVE_FILE)
    print(BACKGROUND_FILE)
    print(GRID_FILE)

    print("\nDone.")


if __name__ == "__main__":
    main()