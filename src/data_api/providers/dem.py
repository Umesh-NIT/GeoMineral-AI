import io

import numpy as np
import requests
import rasterio


PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"


class CopernicusDEM:

    def __init__(self, dem_instance="COPERNICUS_90"):
        self.dem_instance = dem_instance

    def get_elevation(
        self,
        latitude,
        longitude,
        size=5,
        resolution=0.001,
    ):

        half = resolution * size / 2

        bbox = [
            longitude - half,
            latitude - half,
            longitude + half,
            latitude + half,
        ]

        evalscript = """
        //VERSION=3

        function setup() {
            return {
                input: ["DEM"],
                output: {
                    id: "default",
                    bands: 1,
                    sampleType: SampleType.FLOAT32
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.DEM];
        }
        """

        request = {
            "input": {
                "bounds": {
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                    },
                    "bbox": bbox,
                },
                "data": [
                    {
                        "type": "dem",
                        "dataFilter": {
                            "demInstance": self.dem_instance
                        },
                        "processing": {
                            "upsampling": "BILINEAR",
                            "downsampling": "BILINEAR",
                            "egm": False,
                        },
                    }
                ],
            },
            "output": {
                "width": size,
                "height": size,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/tiff"
                        },
                    }
                ],
            },
            "evalscript": evalscript,
        }

        response = requests.post(
            PROCESS_URL,
            json=request,
            timeout=120,
        )

        response.raise_for_status()

        with rasterio.open(
            io.BytesIO(response.content)
        ) as dataset:

            data = dataset.read(1)

        data = data.astype(float)

        data[data <= -9990] = np.nan

        return data