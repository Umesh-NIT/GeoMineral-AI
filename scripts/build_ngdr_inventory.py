import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "ngdr_exports" / "Export.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

INVENTORY_FILE = OUTPUT_DIR / "ngdr_inventory.csv"
LITHIUM_REE_FILE = OUTPUT_DIR / "ngdr_lithium_ree_records.csv"


def load_records():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Export.json must contain a list of records.")

    return records


def build_inventory(records):
    rows = []

    for record in records:
        rows.append(
            {
                "id": record.get("id"),
                "exp_upid": record.get("exp_upid"),
                "commodity": record.get("commodity"),
                "project_title": record.get("project_title"),
                "exploration_agency": record.get(
                    "name_of_exploration_agency"
                ),
                "exploration_stage": record.get("exploration_stage"),
                "state": record.get("state_name"),
                "district": record.get("district_name"),
                "block": record.get("block_name"),
                "toposheet_number": record.get("toposheet_number"),
                "geographical_extent": record.get("geographical_extent"),
                "latitude": record.get("exp_dd_latitude"),
                "longitude": record.get("exp_dd_longitude"),
                "gis_file": record.get(
                    "exploration_georeferenced_gis"
                ),
                "table_file": record.get(
                    "exploration_tubular_and_other_data_file_name"
                ),
                "report_file": record.get("exp_rep_doc_file"),
                "image_file": record.get(
                    "non_georeferenced_images_file_name"
                ),
                "additional_file": record.get(
                    "exploration_non_georeferenced_fileupload"
                ),
                "map_image_file": record.get(
                    "georeferenced_maps_and_images_file_name"
                ),
            }
        )

    return pd.DataFrame(rows)


def filter_lithium_ree(df):
    commodity = df["commodity"].fillna("").astype(str).str.lower()

    lithium_mask = commodity.str.contains("lithium", na=False)

    ree_mask = commodity.str.contains(
        r"\bree\b|rare earth|rare-earth|rare earth elements",
        case=False,
        regex=True,
        na=False,
    )

    result = df[lithium_mask | ree_mask].copy()

    result["target_lithium"] = lithium_mask.loc[result.index].astype(int)
    result["target_ree"] = ree_mask.loc[result.index].astype(int)

    return result


def main():
    print("=" * 70)
    print("GeoMineral AI - NGDR Inventory Builder")
    print("=" * 70)

    print(f"\nInput file:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"NGDR export not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading NGDR records...")
    records = load_records()

    print(f"Total records loaded: {len(records)}")

    print("\nBuilding inventory...")
    inventory = build_inventory(records)

    inventory.to_csv(
        INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Inventory saved:")
    print(INVENTORY_FILE)

    print("\nFiltering Lithium and REE records...")
    target_records = filter_lithium_ree(inventory)

    target_records.to_csv(
        LITHIUM_REE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Target records saved:")
    print(LITHIUM_REE_FILE)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total NGDR records: {len(inventory)}")
    print(
        f"Lithium records: "
        f"{target_records['target_lithium'].sum()}"
    )
    print(
        f"REE records: "
        f"{target_records['target_ree'].sum()}"
    )
    print(
        f"Lithium/REE-related records: "
        f"{len(target_records)}"
    )

    print("\nTarget records by state:")

    if not target_records.empty:
        print(
            target_records["state"]
            .fillna("Unknown")
            .value_counts()
            .to_string()
        )

    print("\nDone.")


if __name__ == "__main__":
    main()