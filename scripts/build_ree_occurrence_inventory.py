import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "ngdr_exports" / "Export.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "occurrences"
OUTPUT_FILE = OUTPUT_DIR / "ree_occurrences.csv"


REE_PATTERN = (
    r"\bree\b|"
    r"rare\s*earth|"
    r"rare-earth|"
    r"rare\s*earth\s*elements"
)


def load_ngdr():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Export.json must contain a list of records.")

    return pd.DataFrame(records)


def build_inventory(df):
    commodity = df["commodity"].fillna("").astype(str)
    title = df["project_title"].fillna("").astype(str)
    keywords = df["exploration_search_keywords"].fillna("").astype(str)

    commodity_ree = commodity.str.contains(
        REE_PATTERN,
        case=False,
        regex=True,
        na=False,
    )

    title_ree = title.str.contains(
        REE_PATTERN,
        case=False,
        regex=True,
        na=False,
    )

    keyword_ree = keywords.str.contains(
        REE_PATTERN,
        case=False,
        regex=True,
        na=False,
    )

    result = df.loc[
        commodity_ree | title_ree | keyword_ree
    ].copy()

    result["ree_commodity_evidence"] = commodity_ree.loc[
        result.index
    ].astype(int)

    result["ree_title_evidence"] = title_ree.loc[
        result.index
    ].astype(int)

    result["ree_keyword_evidence"] = keyword_ree.loc[
        result.index
    ].astype(int)

    result["ree_evidence_score"] = (
        result["ree_commodity_evidence"]
        + result["ree_title_evidence"]
        + result["ree_keyword_evidence"]
    )

    result["has_coordinates"] = (
        pd.to_numeric(
            result["exp_dd_latitude"],
            errors="coerce",
        ).notna()
        &
        pd.to_numeric(
            result["exp_dd_longitude"],
            errors="coerce",
        ).notna()
    ).astype(int)

    result["latitude"] = pd.to_numeric(
        result["exp_dd_latitude"],
        errors="coerce",
    )

    result["longitude"] = pd.to_numeric(
        result["exp_dd_longitude"],
        errors="coerce",
    )

    return result


def main():
    print("=" * 80)
    print("GeoMineral AI - REE Occurrence Inventory")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"NGDR export not found: {INPUT_FILE}"
        )

    print(f"\nInput: {INPUT_FILE}")

    df = load_ngdr()

    print(f"Total NGDR records: {len(df)}")

    ree = build_inventory(df)

    print(f"REE candidate records: {len(ree)}")

    print("\nREE evidence score:")
    print(
        ree["ree_evidence_score"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCoordinate availability:")
    print(
        ree["has_coordinates"]
        .value_counts()
        .rename(
            {
                0: "missing_coordinates",
                1: "valid_coordinates",
            }
        )
        .to_string()
    )

    print("\nExploration stage:")
    print(
        ree["exploration_stage"]
        .fillna("Unknown")
        .value_counts()
        .to_string()
    )

    print("\nTop states:")
    print(
        ree["state_name"]
        .fillna("Unknown")
        .value_counts()
        .head(20)
        .to_string()
    )

    columns = [
        "id",
        "exp_upid",
        "project_title",
        "commodity",
        "exploration_search_keywords",
        "name_of_exploration_agency",
        "exploration_stage",
        "state_name",
        "district_name",
        "block_name",
        "toposheet_number",
        "latitude",
        "longitude",
        "geographical_extent",
        "exploration_georeferenced_gis",
        "exploration_tubular_and_other_data_file_name",
        "exp_rep_doc_file",
        "ree_commodity_evidence",
        "ree_title_evidence",
        "ree_keyword_evidence",
        "ree_evidence_score",
        "has_coordinates",
    ]

    output = ree[columns].copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\nFinal shape:")
    print(output.shape)

    print("\nDone.")


if __name__ == "__main__":
    main()