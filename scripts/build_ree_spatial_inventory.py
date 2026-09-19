import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "ngdr_exports" / "Export.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "occurrences"

PROJECTS_FILE = OUTPUT_DIR / "ree_projects.csv"
POINTS_FILE = OUTPUT_DIR / "ree_occurrence_points.csv"
QC_FILE = OUTPUT_DIR / "ree_coordinate_qc.csv"


REE_PATTERN = (
    r"\bree\b|"
    r"rare\s*earth|"
    r"rare-earth|"
    r"rare\s*earth\s*elements"
)


def split_values(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def parse_dd(value):
    values = split_values(value)

    result = []

    for item in values:
        item = item.replace("\t", "").strip()

        try:
            result.append(float(item))
        except ValueError:
            result.append(None)

    return result


def parse_dms(value):
    values = split_values(value)

    result = []

    pattern = re.compile(
        r"^\s*(\d+(?:\.\d+)?)\s*°\s*"
        r"(\d+(?:\.\d+)?)?\s*['’]?\s*"
        r"(\d+(?:\.\d+)?)?\s*[\"”]?\s*$"
    )

    for item in values:
        item = item.strip()

        match = pattern.match(item)

        if not match:
            result.append(None)
            continue

        degrees = float(match.group(1))
        minutes = float(match.group(2) or 0)
        seconds = float(match.group(3) or 0)

        decimal = (
            degrees
            + minutes / 60
            + seconds / 3600
        )

        result.append(decimal)

    return result


def valid_latitude(value):
    return value is not None and -90 <= value <= 90


def valid_longitude(value):
    return value is not None and -180 <= value <= 180


def extract_coordinates(row):
    extent = str(
        row.get("geographical_extent", "")
    ).strip().upper()

    lat_raw = row.get("exp_dd_latitude")
    lon_raw = row.get("exp_dd_longitude")

    if extent == "DD":
        latitudes = parse_dd(lat_raw)
        longitudes = parse_dd(lon_raw)

        coordinate_type = "DD"

    elif extent == "DMS":
        latitudes = parse_dms(
            row.get("exp_dms_latitude")
        )

        longitudes = parse_dms(
            row.get("exp_dms_longitude")
        )

        coordinate_type = "DMS"

    else:
        return [], "unsupported_extent"

    if len(latitudes) != len(longitudes):
        return [], "coordinate_count_mismatch"

    points = []

    for index, (lat, lon) in enumerate(
        zip(latitudes, longitudes),
        start=1,
    ):
        if not valid_latitude(lat):
            continue

        if not valid_longitude(lon):
            continue

        points.append(
            {
                "point_number": index,
                "latitude": lat,
                "longitude": lon,
                "coordinate_type": coordinate_type,
            }
        )

    return points, "valid"


def main():

    print("=" * 80)
    print("GeoMineral AI - REE Spatial Inventory Builder")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(f"\nInput: {INPUT_FILE}")

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Export.json must contain a list."
        )

    df = pd.DataFrame(records)

    print(f"Total NGDR records: {len(df)}")

    commodity = (
        df["commodity"]
        .fillna("")
        .astype(str)
    )

    title = (
        df["project_title"]
        .fillna("")
        .astype(str)
    )

    keywords = (
        df["exploration_search_keywords"]
        .fillna("")
        .astype(str)
    )

    ree_mask = (
        commodity.str.contains(
            REE_PATTERN,
            case=False,
            regex=True,
            na=False,
        )
        |
        title.str.contains(
            REE_PATTERN,
            case=False,
            regex=True,
            na=False,
        )
        |
        keywords.str.contains(
            REE_PATTERN,
            case=False,
            regex=True,
            na=False,
        )
    )

    ree = df.loc[ree_mask].copy()

    print(f"REE candidate projects: {len(ree)}")

    project_rows = []
    point_rows = []
    qc_rows = []

    for _, row in ree.iterrows():

        project_id = row.get("id")

        points, status = extract_coordinates(row)

        qc_rows.append(
            {
                "project_id": project_id,
                "project_title": row.get(
                    "project_title"
                ),
                "geographical_extent": row.get(
                    "geographical_extent"
                ),
                "coordinate_status": status,
                "coordinate_points": len(points),
            }
        )

        project_rows.append(
            {
                "project_id": project_id,
                "exp_upid": row.get("exp_upid"),
                "project_title": row.get(
                    "project_title"
                ),
                "commodity": row.get(
                    "commodity"
                ),
                "exploration_search_keywords": row.get(
                    "exploration_search_keywords"
                ),
                "exploration_agency": row.get(
                    "name_of_exploration_agency"
                ),
                "exploration_stage": row.get(
                    "exploration_stage"
                ),
                "state": row.get(
                    "state_name"
                ),
                "district": row.get(
                    "district_name"
                ),
                "block": row.get(
                    "block_name"
                ),
                "toposheet_number": row.get(
                    "toposheet_number"
                ),
                "geographical_extent": row.get(
                    "geographical_extent"
                ),
                "gis_file": row.get(
                    "exploration_georeferenced_gis"
                ),
                "table_file": row.get(
                    "exploration_tubular_and_other_data_file_name"
                ),
                "report_file": row.get(
                    "exp_rep_doc_file"
                ),
                "ree_commodity_evidence": int(
                    bool(
                        REE_PATTERN
                    )
                    and bool(
                        re.search(
                            REE_PATTERN,
                            str(
                                row.get(
                                    "commodity",
                                    ""
                                )
                            ),
                            flags=re.I,
                        )
                    )
                ),
                "ree_title_evidence": int(
                    bool(
                        re.search(
                            REE_PATTERN,
                            str(
                                row.get(
                                    "project_title",
                                    ""
                                )
                            ),
                            flags=re.I,
                        )
                    )
                ),
                "ree_keyword_evidence": int(
                    bool(
                        re.search(
                            REE_PATTERN,
                            str(
                                row.get(
                                    "exploration_search_keywords",
                                    ""
                                )
                            ),
                            flags=re.I,
                        )
                    )
                ),
            }
        )

        for point in points:

            point_rows.append(
                {
                    "project_id": project_id,
                    "exp_upid": row.get(
                        "exp_upid"
                    ),
                    "project_title": row.get(
                        "project_title"
                    ),
                    "state": row.get(
                        "state_name"
                    ),
                    "district": row.get(
                        "district_name"
                    ),
                    "exploration_stage": row.get(
                        "exploration_stage"
                    ),
                    "point_number": point[
                        "point_number"
                    ],
                    "latitude": point[
                        "latitude"
                    ],
                    "longitude": point[
                        "longitude"
                    ],
                    "coordinate_type": point[
                        "coordinate_type"
                    ],
                }
            )

    projects = pd.DataFrame(project_rows)

    points = pd.DataFrame(point_rows)

    qc = pd.DataFrame(qc_rows)

    if not points.empty:

        before = len(points)

        points = points.drop_duplicates(
            subset=[
                "project_id",
                "latitude",
                "longitude",
            ]
        )

        print(
            f"\nDuplicate spatial points removed: "
            f"{before - len(points)}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    projects.to_csv(
        PROJECTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    points.to_csv(
        POINTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    qc.to_csv(
        QC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("SPATIAL INVENTORY SUMMARY")
    print("=" * 80)

    print(
        f"\nREE projects: "
        f"{len(projects)}"
    )

    print(
        f"Spatial coordinate points: "
        f"{len(points)}"
    )

    print("\nCoordinate status:")

    print(
        qc["coordinate_status"]
        .value_counts()
        .to_string()
    )

    print("\nCoordinate types:")

    if not points.empty:

        print(
            points["coordinate_type"]
            .value_counts()
            .to_string()
        )

        print("\nCoordinate range:")

        print(
            f"Latitude: "
            f"{points['latitude'].min()} "
            f"to "
            f"{points['latitude'].max()}"
        )

        print(
            f"Longitude: "
            f"{points['longitude'].min()} "
            f"to "
            f"{points['longitude'].max()}"
        )

    print("\nSaved:")

    print(PROJECTS_FILE)
    print(POINTS_FILE)
    print(QC_FILE)

    print("\nFinal shapes:")

    print(
        f"Projects: {projects.shape}"
    )

    print(
        f"Points: {points.shape}"
    )

    print(
        f"QC: {qc.shape}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()