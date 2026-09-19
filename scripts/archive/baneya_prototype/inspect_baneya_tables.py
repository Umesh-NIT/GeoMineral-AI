from pathlib import Path
import pandas as pd

BASE = Path("data/interim/ngdr/baneya/tables_extracted/Tables")

files = [
    BASE / "Chemical_results_50753.xlsx",
    BASE / "Sample_location_50753.xls",
    BASE / "Oriented_Structure_Plane_LSM.xls",
]

for path in files:
    print("\n" + "=" * 100)
    print(f"FILE: {path.name}")
    print("=" * 100)

    try:
        excel = pd.ExcelFile(path)
        print("Sheets:", excel.sheet_names)

        for sheet in excel.sheet_names:
            print("\n" + "-" * 100)
            print(f"SHEET: {sheet}")
            print("-" * 100)

            df = pd.read_excel(path, sheet_name=sheet)

            print("Shape:", df.shape)
            print("Columns:")
            for i, column in enumerate(df.columns, 1):
                print(f"  {i}. {column}")

            print("\nFirst 5 rows:")
            print(df.head().to_string(index=False))

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")