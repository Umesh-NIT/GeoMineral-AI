from pathlib import Path
import pandas as pd


INPUT = Path(
    "data/interim/ngdr/baneya/tables_extracted/Tables/Oriented_Structure_Plane_LSM.xls"
)


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Structural Data Inspection")
    print("=" * 100)

    xls = pd.ExcelFile(INPUT)

    print("\nSheets:")
    print(xls.sheet_names)

    for sheet in xls.sheet_names:

        print("\n" + "=" * 100)
        print(f"SHEET: {sheet}")
        print("=" * 100)

        df = pd.read_excel(
            INPUT,
            sheet_name=sheet
        )

        print("\nShape:")
        print(df.shape)

        print("\nColumns:")
        for i, column in enumerate(df.columns, 1):
            print(f"{i:02d}. {column}")

        print("\nMissing values:")
        print(
            df.isna()
            .sum()
            .sort_values(ascending=False)
            .to_string()
        )

        print("\nUnique POINT_TYPE:")
        if "POINT_TYPE" in df.columns:
            print(
                df["POINT_TYPE"]
                .value_counts(dropna=False)
                .to_string()
            )

        print("\nCoordinate range:")

        if "Latitude" in df.columns:
            print(
                "Latitude:",
                df["Latitude"].min(),
                "to",
                df["Latitude"].max()
            )

        if "Longitude" in df.columns:
            print(
                "Longitude:",
                df["Longitude"].min(),
                "to",
                df["Longitude"].max()
            )

        print("\nStrike statistics:")

        if "STRIKE" in df.columns:
            print(
                df["STRIKE"]
                .describe()
                .to_string()
            )

        print("\nDip statistics:")

        if "DIP" in df.columns:
            print(
                df["DIP"]
                .describe()
                .to_string()
            )

        print("\nFirst 20 records:")
        print(
            df.head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()