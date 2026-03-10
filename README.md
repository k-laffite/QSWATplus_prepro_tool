# QSWAT+ Inputs Preprocessing App

A Streamlit-based desktop/web app for preparing spatial input layers for **QSWAT+**.  
The app guides you through uploading, inspecting, and preprocessing DEM, Land Use, Soil, and HUC boundary data so they are ready to use as inputs for QSWAT+ projects.

---

## Features

- **Step‑by‑step workflow** for building QSWAT+ input layers
- **DEM, Land Use, Soil, HUC** upload and preview
- **Vector → raster conversion** (e.g., land use polygons to raster)
- **CRS harmonization**: reproject all layers to a user‑selected target CRS
- **Mosaic multiple rasters** per layer into a single seamless raster
- **Clip to HUC watershed** (optional)
- **Land Use class extraction** from rasters/vectors
- **Interactive SWAT+ class mapping table**:
  - Assign `SWAT_Class_Target` via dropdown or text
  - Auto‑populate `SWAT_ID` and `SWAT_Description` from a SWAT+ reference CSV
- **Optional Land Use reclassification**:
  - Reclassify the Land Use raster from original codes to `SWAT_ID` values
  - Export a clean SWAT+ land use lookup table
- **Final ZIP export** of all processed layers and lookup tables

---

## Project Structure (high level)

- `app.py` – main Streamlit application, holds the full multi‑step workflow.
- `utils/`
  - `file_handlers.py` – file upload helpers, caching, and I/O utilities.
  - `spatial_processing.py` – reprojection, mosaicking, clipping, raster reclassification, etc.
  - `map_utils.py` – leaflet/folium map creation helpers for previews.
- `data/`
  - `huc_gdb/`, `huc_shapefiles/`, `sample/` – reference/example datasets (if provided).
  - `swat+_classes/` – expected location for the SWAT+ class reference CSV (e.g. `swat+_classes.csv`).
- `.streamlit/config.toml` – Streamlit theme and configuration.

Exact filenames may vary slightly; the important concepts are the **app**, **spatial utils**, and **reference data**.

---

## Installation

### 1. Create and activate a Python environment

# Example with conda
conda create -n qswat_inputs python=3.11 -y
conda activate qswat_inputs
