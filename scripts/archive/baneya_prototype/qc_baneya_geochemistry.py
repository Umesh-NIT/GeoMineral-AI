from pathlib import Path
import pandas as pd


INPUT = Path(
    "data/processed/geochemistry/baneya_geochemical_master.csv"
)

OUTPUT = Path(
    "data/processed/geochemistry/baneya_geochemical_qc.csv"
)


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Geochemical Quality Control")
    print("=" * 100)

    df = pd.read_csv(INPUT)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    for i, column in enumerate(df.columns, 1):
        print(f"{i:02d}. {column}")

    print("\n" + "=" * 100)
    print("SAMPLE CATEGORY")
    print("=" * 100)

    print(
        df["sample_category"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n" + "=" * 100)
    print("MISSING VALUES")
    print("=" * 100)

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing[missing > 0].to_string())

    print("\n" + "=" * 100)
    print("COORDINATE CHECK")
    print("=" * 100)

    print(
        "Latitude missing:",
        df["latitude"].isna().sum()
    )

    print(
        "Longitude missing:",
        df["longitude"].isna().sum()
    )

    print(
        "Latitude range:",
        df["latitude"].min(),
        "to",
        df["latitude"].max()
    )

    print(
        "Longitude range:",
        df["longitude"].min(),
        "to",
        df["longitude"].max()
    )

    print("\n" + "=" * 100)
    print("LITHIUM / REE COLUMNS")
    print("=" * 100)

    target_keywords = [
        "li",
        "lithium",
        "la",
        "ce",
        "pr",
        "nd",
        "sm",
        "eu",
        "gd",
        "tb",
        "dy",
        "ho",
        "er",
        "tm",
        "yb",
        "lu",
        "ree",
        "cs",
        "rb",
        "nb",
        "ta",
        "sn",
        "th",
        "u",
    ]

    target_columns = []

    for column in df.columns:

        name = str(column).strip().lower()

        if any(keyword == name or keyword in name for keyword in target_keywords):
            target_columns.append(column)

    for column in target_columns:

        print("\n" + "-" * 80)
        print(f"COLUMN: {column}")
        print("-" * 80)

        print("Data type:", df[column].dtype)
        print("Non-null:", df[column].notna().sum())
        print("Missing:", df[column].isna().sum())

        print("Unique values:", df[column].nunique(dropna=True))

        print("Sample values:")

        print(
            df[column]
            .dropna()
            .astype(str)
            .head(20)
            .to_string(index=False)
        )

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if numeric.notna().sum() > 0:

            print("Numeric values:", numeric.notna().sum())
            print("Minimum:", numeric.min())
            print("Maximum:", numeric.max())
            print("Mean:", numeric.mean())
            print("Median:", numeric.median())

    print("\n" + "=" * 100)
    print("DUPLICATE SAMPLE IDs")
    print("=" * 100)

    duplicates = df[
        df["sample_id"].duplicated(
            keep=False
        )
    ]

    print(
        "Duplicate rows:",
        len(duplicates)
    )

    if not duplicates.empty:
        print(
            duplicates[
                [
                    "sample_id",
                    "sample_category",
                    "latitude",
                    "longitude",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 100)
    print("EXACT DUPLICATE ROWS")
    print("=" * 100)

    print(
        "Exact duplicate rows:",
        df.duplicated().sum()
    )

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()