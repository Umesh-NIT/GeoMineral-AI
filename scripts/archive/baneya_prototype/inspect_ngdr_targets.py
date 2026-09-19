from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ngdr_lithium_ree_records.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "selected_projects.csv"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    df["commodity"] = df["commodity"].fillna("").astype(str)
    df["project_title"] = df["project_title"].fillna("").astype(str)
    df["state"] = df["state"].fillna("").astype(str)
    df["district"] = df["district"].fillna("").astype(str)
    df["exploration_stage"] = (
        df["exploration_stage"].fillna("").astype(str)
    )

    lithium = df[df["target_lithium"] == 1].copy()

    lithium["has_gis"] = (
        lithium["gis_file"].fillna("").astype(str).str.strip().ne("")
    )

    lithium["has_tables"] = (
        lithium["table_file"].fillna("").astype(str).str.strip().ne("")
    )

    lithium["has_report"] = (
        lithium["report_file"].fillna("").astype(str).str.strip().ne("")
    )

    lithium["has_coordinates"] = (
        lithium["latitude"].fillna("").astype(str).str.strip().ne("")
        & lithium["longitude"].fillna("").astype(str).str.strip().ne("")
    )

    lithium["data_score"] = (
        lithium["has_gis"].astype(int)
        + lithium["has_tables"].astype(int)
        + lithium["has_report"].astype(int)
        + lithium["has_coordinates"].astype(int)
    )

    columns = [
        "id",
        "exp_upid",
        "commodity",
        "project_title",
        "exploration_agency",
        "exploration_stage",
        "state",
        "district",
        "block",
        "latitude",
        "longitude",
        "gis_file",
        "table_file",
        "report_file",
        "has_gis",
        "has_tables",
        "has_report",
        "has_coordinates",
        "data_score",
    ]

    lithium = lithium[columns].sort_values(
        by=["data_score", "exploration_stage"],
        ascending=[False, True],
    )

    lithium.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("GeoMineral AI - NGDR Lithium Target Inspection")
    print("=" * 80)

    print(f"\nTotal Lithium records: {len(lithium)}")

    print("\nData availability:")
    print(f"GIS files: {lithium['has_gis'].sum()}")
    print(f"Tables: {lithium['has_tables'].sum()}")
    print(f"Reports: {lithium['has_report'].sum()}")
    print(f"Coordinates: {lithium['has_coordinates'].sum()}")

    print("\nLithium projects:")
    print(
        lithium[
            [
                "id",
                "project_title",
                "exploration_stage",
                "state",
                "district",
                "data_score",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()