from datetime import date
from typing import Any

import requests
from pystac_client import Client


STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

SENTINEL2_COLLECTION = "sentinel-2-l2a"


class CopernicusSTAC:

    def __init__(self, url: str = STAC_URL):
        self.client = Client.open(url)

    def search_sentinel2(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        cloud_cover: float = 20.0,
        max_items: int = 10,
    ) -> list[dict[str, Any]]:

        delta = 0.01

        bbox = [
            longitude - delta,
            latitude - delta,
            longitude + delta,
            latitude + delta,
        ]

        search = self.client.search(
            collections=[SENTINEL2_COLLECTION],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={
                "eo:cloud_cover": {
                    "lte": cloud_cover
                }
            },
            max_items=max_items,
        )

        results = []

        for item in search.items():

            results.append(
                {
                    "id": item.id,
                    "datetime": item.datetime.isoformat()
                    if item.datetime
                    else None,
                    "cloud_cover": item.properties.get(
                        "eo:cloud_cover"
                    ),
                    "collection": item.collection_id,
                    "bbox": item.bbox,
                    "assets": {
                        name: asset.href
                        for name, asset in item.assets.items()
                    },
                }
            )

        return results


def check_endpoint() -> bool:

    response = requests.get(
        STAC_URL,
        timeout=30,
    )

    response.raise_for_status()

    return True