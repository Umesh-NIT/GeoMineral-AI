from pathlib import Path
import pandas as pd
import numpy as np


INPUT = Path(
    "data/processed/geochemistry/baneya_geochemical_master.csv"
)

OUTPUT = Path(
    "data/processed/geochemistry/baneya_geochemical_standardized.csv"
)

QC_OUTPUT = Path(
    "data/processed/geochemistry/baneya_detection_limits.csv"
)


CHEMICAL_COLUMNS = [
    "Cu",
    "Pb",
    "Zn",
    "Li",
    "Ag",
    "Be",
    "Sc*",
    "As",
    "Rb",
    "Y*",
    "Nb*",
    "Sn",
    "Cs*",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Eu",
    "Sm",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Ta",
    "W*",
    "Th",
    "U",
    "Au",
    "Nb+Ta",
    "Cu(ppm)",
    "Pb(ppm)",
    "Zn(ppm)",
    "Ni(ppm)",
    "Co(ppm)",
    "Cr(ppm)",
    "Bi",
    "Tl",
    "V*",
    "Ga*",
    "Ge",
    "Sr*",
    "Ba*",
    "SiO2",
    "Al2O3",
    "Fe2O3T",
    "CaO",
    "MgO",
    "Na2O",
    "K2O",
    "TiO2",
    "MnO",
    "P2O5",
    "LOI",
    "Co",
    "Ni",
    "Cr",
    "Zr*",
]


def convert_value(value):

    if pd.isna(value):
        return np.nan, False, np.nan

    text = str(value).strip()

    if not text:
        return np.nan, False, np.nan

    if text.startswith("<"):
        detection_text = text[1:].strip()

        try:
            detection_limit = float(detection_text)
            return detection_limit, True, detection_limit
        except ValueError:
            return np.nan, True, np.nan

    if text.startswith(">"):
        numeric_text = text[1:].strip()

        try:
            return float(numeric_text), False, np.nan
        except ValueError:
            return np.nan, False, np.nan

    try:
        return float(text), False, np.nan
    except ValueError:
        return np.nan, False, np.nan


def main():

    print("=" * 100)
    print("GeoMineral AI - Baneya Chemistry Standardization")
    print("=" * 100)

    df = pd.read_csv(INPUT)

    print("\nInput shape:", df.shape)

    detection_records = []

    for column in CHEMICAL_COLUMNS:

        if column not in df.columns:
            continue

        values = []
        censored = []
        limits = []

        for value in df[column]:

            numeric_value, is_censored, detection_limit = convert_value(
                value
            )

            values.append(numeric_value)
            censored.append(is_censored)
            limits.append(detection_limit)

        df[f"{column}_value"] = values
        df[f"{column}_censored"] = censored

        censored_count = sum(censored)

        if censored_count > 0:

            detection_records.append(
                {
                    "column": column,
                    "censored_count": censored_count,
                    "detection_limits": sorted(
                        {
                            x
                            for x in limits
                            if not pd.isna(x)
                        }
                    ),
                }
            )

    detection_df = pd.DataFrame(detection_records)

    df.to_csv(
        OUTPUT,
        index=False
    )

    detection_df.to_csv(
        QC_OUTPUT,
        index=False
    )

    print("\n" + "=" * 100)
    print("STANDARDIZATION SUMMARY")
    print("=" * 100)

    print("\nInput rows:", len(pd.read_csv(INPUT)))
    print("Output rows:", len(df))
    print("Output columns:", len(df.columns))

    print("\nLithium:")
    print(
        "Raw non-null:",
        df["Li"].notna().sum()
    )

    print(
        "Numeric Li values:",
        df["Li_value"].notna().sum()
    )

    print(
        "Li censored values:",
        df["Li_censored"].sum()
    )

    print("\nLithium value statistics:")

    print(
        df["Li_value"]
        .describe()
        .to_string()
    )

    print("\nDetection-limit summary:")

    if detection_df.empty:
        print("No detection-limit values found.")
    else:
        print(
            detection_df.to_string(index=False)
        )

    print("\nSaved:")
    print(OUTPUT)
    print(QC_OUTPUT)


if __name__ == "__main__":
    main()