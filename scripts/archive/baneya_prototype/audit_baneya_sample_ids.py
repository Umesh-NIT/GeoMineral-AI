from pathlib import Path
import pandas as pd


BASE = Path("data/interim/ngdr/baneya/tables_extracted/Tables")

CHEMICAL_FILE = BASE / "Chemical_results_50753.xlsx"
LOCATION_FILE = BASE / "Sample_location_50753.xls"


def clean_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if not value:
        return None

    return value


def main():
    print("=" * 100)
    print("GeoMineral AI - Baneya Sample ID Audit")
    print("=" * 100)

    locations = pd.read_excel(
        LOCATION_FILE,
        sheet_name="64N10_Chem_samples"
    )

    locations["SAMPLE_NUMBER"] = locations["SAMPLE_NUMBER"].apply(clean_id)

    location_ids = set(
        locations["SAMPLE_NUMBER"]
        .dropna()
        .tolist()
    )

    print("\nSample-location records:", len(locations))
    print("Unique location sample IDs:", len(location_ids))

    print("\n" + "=" * 100)
    print("CHEMICAL SAMPLE IDS")
    print("=" * 100)

    excel = pd.ExcelFile(CHEMICAL_FILE)

    all_chemical_ids = []

    for sheet in excel.sheet_names:

        df = pd.read_excel(
            CHEMICAL_FILE,
            sheet_name=sheet
        )

        sample_column = None

        for column in df.columns:
            name = str(column).strip().lower()

            if name in {"sample", "sample number", "sample no."}:
                sample_column = column
                break

        print(f"\nSheet: {sheet}")
        print("Rows:", len(df))
        print("Sample column:", sample_column)

        if sample_column is None:
            print("WARNING: No sample identifier column found.")
            continue

        df["sample_id_clean"] = df[sample_column].apply(clean_id)

        ids = set(
            df["sample_id_clean"]
            .dropna()
            .tolist()
        )

        matched = ids.intersection(location_ids)
        unmatched = ids - location_ids

        print("Unique chemical IDs:", len(ids))
        print("IDs matching location table:", len(matched))
        print("IDs NOT matching location table:", len(unmatched))

        print("\nFirst 20 chemical IDs:")
        for value in sorted(ids)[:20]:
            print(" ", value)

        if unmatched:
            print("\nFirst 20 unmatched IDs:")
            for value in sorted(unmatched)[:20]:
                print(" ", value)

        all_chemical_ids.extend(
            [(sheet, value) for value in ids]
        )

    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    unique_chemical_ids = set(
        value for _, value in all_chemical_ids
    )

    matched_overall = unique_chemical_ids.intersection(location_ids)

    print("Unique chemical sample IDs:", len(unique_chemical_ids))
    print("Unique location sample IDs:", len(location_ids))
    print("Chemical IDs with coordinates:", len(matched_overall))

    print("\nMatched sample IDs:")
    for value in sorted(matched_overall):
        print(" ", value)


if __name__ == "__main__":
    main()