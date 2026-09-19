from src.data_api.providers.copernicus import (
    CopernicusSTAC,
    check_endpoint,
)


def main():

    print("=" * 70)
    print("GeoMineral AI - Copernicus API Test")
    print("=" * 70)

    print("\nTesting STAC endpoint...")

    if check_endpoint():
        print("STAC endpoint: OK")

    api = CopernicusSTAC()

    print("\nSearching Sentinel-2...")

    results = api.search_sentinel2(
        latitude=22.65,
        longitude=83.59,
        start_date="2025-01-01",
        end_date="2025-12-31",
        cloud_cover=20,
        max_items=5,
    )

    print(
        f"\nSentinel-2 results: {len(results)}"
    )

    for result in results:

        print("\nID:", result["id"])
        print(
            "Date:",
            result["datetime"]
        )
        print(
            "Cloud:",
            result["cloud_cover"]
        )
        print(
            "Assets:",
            list(result["assets"].keys())
        )

    print("\nDone.")


if __name__ == "__main__":
    main()