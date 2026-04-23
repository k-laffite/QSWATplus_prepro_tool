# QSWAT+ Inputs Preprocessor

A **Streamlit** app for building and exporting spatial and tabular inputs for **QSWAT+ (SWAT+ in QGIS)**. It guides you from watershed context through uploads, land-use mapping, optional field-specific soil HRUs, preprocessing options, and a final ZIP of GeoTIFFs and CSV lookups.

---

## What it does

- **DEM, Land Use, and Soil** layers with separate upload areas, map previews, and an **active source summary** (which raster or vector is driving each layer).
- **HUC (watershed) boundary** selection from a File Geodatabase in `data/huc_gdb/`.
- **Vector → raster** conversion for land use and soil (DEM-matched grid, optional merge of multiple shapefiles from a ZIP).
- **Soil rasters with attribute tables (VAT)**: map **pixel/cell values** in the GeoTIFF to **MUKEY** (or equivalent) using a sidecar **CSV/DBF** or an uploaded table—**not** row Object ID.
- **Preprocessing**: target CRS, optional **mosaic** of multiple rasters per layer, **clip to HUC**, and optional **land use reclassification** to `SWAT_ID` with lookup exports.
- **Step 4 (optional)**: **field-specific soil HRUs** using crop field polygons, `data/ssurgo_soil_classes/SSURGO_Soils.csv`, and outputs `soil_field_specific.tif`, `lookup_soil.csv`, and `usersoil.csv` (unique `SNAM` style names aligned with the lookup).
- **Final export**: processed rasters, vectors, and CSVs in a **ZIP** for use in QSWAT+ or elsewhere.

---

## Application workflow (steps)

| Step | Name | Summary |
|------|------|---------|
| **1** | HUC Watershed Selection | Load a HUC polygon from a `.gdb` in `data/huc_gdb/` to support clipping and context. |
| **2** | Uploads and Summary | Upload DEM, land use, and soil (GeoTIFF, shapefiles, ZIPs with many `.shp` files). Per-layer vector tools: merge shapefiles, pick value column, rasterize to DEM. **Soil (raster)**: configure VAT file and choose **MUKEY** vs **pixel value** columns. |
| **3** | Land Use Extraction | Extract classes from the active land use layer; **edit** SWAT+ targets in a table (`data/swat+_classes/swat+_classes.csv` drives dropdowns when present). |
| **4** | Advanced Agricultural HRU (optional) | Upload crop field polygons; align soil to DEM; optional **field–soil overlay**; writes field-specific soil raster and soil lookup / usersoil tables. |
| **5** | Preprocessing Options | Choose **target CRS**, **mosaic** multi-part rasters, **clip to HUC**, and whether to **reclassify land use** to `SWAT_ID`. |
| **6** | Final Preview and Export | Run preprocessing, preview layers on a map, **download a ZIP** of outputs. |

When Step 4 has run, preprocessing uses the **field-specific soil raster** in place of the original soil stack for that export.

---

## Project structure

```text
Cursor_Prepro_app/
├── app.py                 # Main Streamlit app (all steps, session state, UI)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml      # Streamlit server/UI settings
└── utils/
    ├── file_handlers.py   # Loads rasters, vectors, CSV/DBF, ZIPs; table helpers
    ├── spatial_processing.py
    │                      # HUC, mosaic, clip, reproject, rasterize, warp, land
    │                      # use reclass, field-specific soil overlay, lookup/usersoil
    └── map_utils.py       # Map previews (leafmap / folium)
```

### Reference data (expected under `data/`)

| Path | Role |
|------|------|
| `data/huc_gdb/*.gdb` | HUC File Geodatabase(s) for Step 1 |
| `data/swat+_classes/swat+_classes.csv` | Optional; columns like `id`, `code`, `description` for Step 3 SWAT+ class pickers |
| `data/ssurgo_soil_classes/SSURGO_Soils.csv` | Required for **Step 4** field-specific soils: SSURGO-style table used to build `usersoil` and names in `lookup_soil` |

Subfolders `data/huc_gdb/`, `data/ssurgo_soil_classes/`, `data/sample/`, etc. may include their own `README.md` with format notes.

---

## Installation

### 1. Python environment

Example with conda:

```bash
conda create -n qswat_inputs python=3.11 -y
conda activate qswat_inputs
```

### 2. Install dependencies

From the directory that contains `app.py`:

```bash
pip install -r requirements.txt
```

On **Windows**, GeoPandas, Rasterio, and Fiona often install more smoothly from **conda-forge** first. If `pip install` fails, install those from conda-forge, then install the rest from `requirements.txt`.

---

## Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (commonly [http://localhost:8501](http://localhost:8501)).

---

## Using soil rasters and VATs

Many SSURGO or ArcGIS **Lup** rasters store **class codes in pixels** (e.g. 121) while **MUKEY** (e.g. 94477) is only in the **attribute table**. In Step 2:

1. Supply a **CSV or DBF** (sidecar with the same base name as the `.tif`, or upload a table manually).
2. Set **“Column with raster / cell values”** to the field that matches **pixel values in the GeoTIFF** (often *Value* or *UniqueValue*), **not** Object ID.
3. Set **“Column with MUKEY”** to the true map-unit key for SSURGO.

The app builds a **raster value → MUKEY** map and passes it into Step 4 and preprocessing so `lookup_soil.csv` and SSURGO joins stay consistent.

---

## Outputs (typical)

After **Run Preprocessing** / export, the ZIP may include (depending on options):

- Processed **DEM**, **land use** (optionally reclassified), **soil** GeoTIFFs
- `land_use_lookup.csv` (and `swat_landuse_lookup.csv` if reclassifying to `SWAT_ID`)
- `lookup_soil.csv` and `usersoil.csv` when **field-specific soils** (Step 4) were generated

---

## License and attribution

Add your license and credit third-party data (HUC, SSURGO, SWAT+) as required by your project.
