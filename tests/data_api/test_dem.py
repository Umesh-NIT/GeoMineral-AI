import numpy as np

from src.data_api.providers.dem import CopernicusDEM


def main():

    print("=" * 70)
    print("GeoMineral AI - Copernicus DEM API Test")
    print("=" * 70)

    latitude = 22.686019
    longitude = 83.532464

    print(f"\nLatitude : {latitude}")
    print(f"Longitude: {longitude}")

    dem = CopernicusDEM(
        dem_instance="COPERNICUS_90"
    )

    print("\nRequesting DEM...")

    data = dem.get_elevation(
        latitude=latitude,
        longitude=longitude,
        size=5,
        resolution=0.001,
    )

    print("\nDEM request successful.")

    print("Shape:", data.shape)
    print("Valid pixels:", np.isfinite(data).sum())

    print(
        "Minimum elevation:",
        np.nanmin(data)
    )

    print(
        "Maximum elevation:",
        np.nanmax(data)
    )

    print(
        "Mean elevation:",
        np.nanmean(data)
    )

    print(
        "Center elevation:",
        data[data.shape[0] // 2, data.shape[1] // 2]
    )

    print("\nNo DEM file was saved locally.")
    print("Done.")


if __name__ == "__main__":
    main()