# GeoMineral AI — Data Sources

## Purpose

This document records all datasets used by GeoMineral AI for mineral prospectivity mapping.

The initial prediction targets are:

- Lithium
- Rare Earth Elements (REE)

---

## Dataset Inventory

| Dataset | Category | Format | Spatial Type | Purpose | Status |
|---|---|---|---|---|---|
| Mineral Occurrences | Exploration | TBD | Point | Training labels / validation | Pending |
| Geological Map | Geological | TBD | Vector/Raster | Lithology and geological context | Pending |
| Sentinel-2 | Remote Sensing | GeoTIFF | Raster | Spectral features | Pending |
| Aeromagnetic | Geophysics | TBD | Raster | Magnetic features | Pending |
| Geochemistry | Geochemistry | TBD | Point/Table | Elemental features | Pending |
| DEM | Topography | GeoTIFF | Raster | Terrain features | Pending |
| Geological Reports | Knowledge | PDF | Document | RAG knowledge base | Pending |

---

## Required Metadata

For every dataset, record:

- Dataset name
- Provider
- Source URL
- Download/API method
- File format
- Coordinate reference system
- Spatial coverage
- Spatial resolution
- Temporal coverage
- Temporal resolution
- Important attributes/bands
- File size
- License
- Preprocessing requirements
- Quality issues
- Final storage location

---

## Processing Principle

All geospatial datasets used together for ML must be spatially aligned.

Required common properties include:

- Study area
- Coordinate reference system
- Spatial extent
- Spatial resolution/grid where applicable
- No-data handling
- Sampling strategy

---

## ML Data Flow

Raw datasets:

```text
Occurrences
Geology
Sentinel-2
Aeromagnetic
Geochemistry
DEM