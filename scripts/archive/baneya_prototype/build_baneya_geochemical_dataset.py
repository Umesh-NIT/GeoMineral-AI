from pathlib import Path
import pandas as pd


BASE = Path("data/interim/ngdr/baneya/tables_extracted/Tables")
OUTPUT = Path("data/processed/geochemistry")

CHEMICAL_FILE = BASE / "Chemical_results_50753.xlsx"
LOCATION_FILE = BASE / "Sample_location_50753.xls"


def clean_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    return value if value else None


def find_sample_column(df):
    for column in df.columns:
        name = str(column).strip().lower()

        if name in {
            "sample",
            "sample number",
            "sample no.",
            "sample no",
        }:
            return column

    raise ValueError(
        f"No sample identifier column found. Columns: {list(df.columns)}"
    )


def main():

    print("=" * 100)
    print("GeoMineral AI - Building Baneya Geochemical Dataset")
    print("=" * 100)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    locations = pd.read_excel(
        LOCATION_FILE,
        sheet_name="64N10_Chem_samples",
    )

    locations["sample_id"] = locations["SAMPLE_NUMBER"].apply(clean_id)

    location_columns = [
        "sample_id",
        "TOPOSHEET_NO",
        "LATITUDE_DD",
        "LONGITUDE_DD",
        "AREA_BLOCK_NAME",
        "COMMODITY_NAME",
        "SAMPLE_TYPE",
        "SAMPLE_MEDIA",
        "REMARKS",
    ]

    locations = locations[location_columns].copy()

    locations = locations.rename(
        columns={
            "TOPOSHEET_NO": "toposheet_no",
            "LATITUDE_DD": "latitude",
            "LONGITUDE_DD": "longitude",
            "AREA_BLOCK_NAME": "area_block",
            "COMMODITY_NAME": "commodity",
            "SAMPLE_TYPE": "sample_type",
            "SAMPLE_MEDIA": "sample_media",
            "REMARKS": "remarks",
        }
    )

    excel = pd.ExcelFile(CHEMICAL_FILE)

    datasets = []

    for sheet in excel.sheet_names:

        print(f"\nProcessing: {sheet}")

        df = pd.read_excel(
            CHEMICAL_FILE,
            sheet_name=sheet,
        )

        sample_column = find_sample_column(df)

        df["sample_id"] = df[sample_column].apply(clean_id)

        df["sample_category"] = sheet

        df = df.drop(columns=[sample_column])

        datasets.append(df)

        print("Rows:", len(df))
        print("Sample column:", sample_column)

    chemistry = pd.concat(
        datasets,
        ignore_index=True,
        sort=False,
    )

    print("\nTotal chemical rows:", len(chemistry))

    chemistry_with_id = chemistry[
        chemistry["sample_id"].notna()
    ].copy()

    locations_with_id = locations[
        locations["sample_id"].notna()
    ].copy()

    duplicate_chemical_ids = (
        chemistry_with_id[
            chemistry_with_id["sample_id"].duplicated(keep=False)
        ]
        .sort_values("sample_id")
    )

    if not duplicate_chemical_ids.empty:
        print("\nWARNING: Duplicate chemical sample IDs found:")
        print(
            duplicate_chemical_ids[
                ["sample_id", "sample_category"]
            ].to_string(index=False)
        )

    master = locations_with_id.merge(
        chemistry_with_id,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    print("Coordinate-linked chemical samples:", len(master))

    master.to_csv(
        OUTPUT / "baneya_geochemical_master.csv",
        index=False,
    )

    unmatched = chemistry_with_id[
    ~chemistry_with_id["sample_id"].isin(
        locations_with_id["sample_id"]
    )
    ].copy()

    unmatched.to_csv(
        OUTPUT / "baneya_unmatched_chemical_samples.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        OUTPUT / "baneya_geochemical_master.csv"
    )

    print(
        OUTPUT / "baneya_unmatched_chemical_samples.csv"
    )

    print("\nRows by sample category:")

    print(
        master["sample_category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nColumns:", len(master.columns))


if __name__ == "__main__":
    main()