"""
QSWAT+ Inputs Preprocessor — Streamlit app.

Sequential workflow:
  1. HUC Watershed Selection
  2. Uploads and Summary (DEM, Land Use, Soil)
  3. Land Use Class Extraction & Export
  4. Preprocessing Options (CRS, mosaic, clip, reclassification CSV)
  5. Final Preview and Export
"""

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import rasterio
import streamlit as st
import streamlit.components.v1 as components

from utils.file_handlers import (
    load_raster,
    load_vector_from_path,
    load_vector_from_zip,
)
from utils.map_utils import create_multi_layer_map
from utils.spatial_processing import (
    ensure_huc_gdb_dir,
    find_and_load_huc,
    get_raster_resolution,
    mosaic_rasters,
    clip_raster_to_geometry,
    clip_vector_to_geometry,
    reproject_raster,
    reproject_vector,
    extract_landuse_classes_raster,
    extract_landuse_classes_vector,
    reclassify_raster_from_lookup,
)

# Project root (directory containing app.py)
PROJECT_ROOT = Path(__file__).resolve().parent


def init_page_config() -> None:
    """Configure Streamlit page."""
    st.set_page_config(
        page_title="QSWAT+ Inputs Preprocessor",
        layout="wide",
        page_icon="🌊",
    )


def init_session_state() -> None:
    """Create data/huc_gdb and initialize all session state keys."""
    ensure_huc_gdb_dir(PROJECT_ROOT)

    defaults = {
        "huc_boundary": None,
        "huc_metadata": None,
        "dem_uploads": [],
        "landuse_uploads": [],
        "soil_uploads": [],
        "land_use_classes_df": None,
        "reclass_lookup_df": None,
        "target_crs": "EPSG:4326",
        "do_mosaic": False,
        "clip_to_huc": False,
        "processed_outputs": None,
        "upload_cache_dir": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state["upload_cache_dir"] is None:
        cache = Path(tempfile.mkdtemp(prefix="qswat_upload_"))
        st.session_state["upload_cache_dir"] = cache


def _save_uploaded_files(
    uploaded_files: List[Any],
    layer_key: str,
    accepted_extensions: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    """
    Save uploaded files to cache and return list of {path, name, metadata}.
    Handles .tif/.tiff and .zip (shapefile).
    """
    if not uploaded_files:
        return []

    cache_dir = Path(st.session_state["upload_cache_dir"])
    layer_dir = cache_dir / layer_key
    layer_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous files for this layer so we replace with new selection
    for f in layer_dir.iterdir():
        try:
            f.unlink()
        except OSError:
            pass

    results = []
    for i, uf in enumerate(uploaded_files):
        name = uf.name
        suffix = Path(name).suffix.lower()
        if suffix not in accepted_extensions:
            continue
        path = layer_dir / f"{i}_{name}"
        path.write_bytes(uf.getbuffer())

        try:
            if suffix == ".zip":
                with zipfile.ZipFile(path, "r") as zf:
                    extract_dir = layer_dir / f"{i}_extracted"
                    extract_dir.mkdir(exist_ok=True)
                    zf.extractall(extract_dir)
                gdf, meta = load_vector_from_path(extract_dir)
                meta["path"] = str(extract_dir)
                meta["name"] = name
                meta["resolution"] = None
                results.append({"path": str(extract_dir), "name": name, "metadata": meta, "gdf": gdf})
            elif suffix in (".tif", ".tiff"):
                _, meta = load_raster(path)
                meta["name"] = name
                results.append({"path": str(path), "name": name, "metadata": meta})
            else:
                continue
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
            continue

    return results


def _get_layer_data_for_map(entry: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Return (data, metadata) for one layer entry for mapping. Data is gdf or raster path."""
    meta = entry["metadata"]
    if meta["type"] == "vector":
        gdf, _ = load_vector_from_path(Path(entry["path"]))
        return gdf, meta
    return entry["path"], meta


def step1_huc_section() -> None:
    """Step 1: HUC Watershed Selection."""
    st.header("Step 1: HUC Watershed Selection")
    st.markdown(
        "Optionally enter a HUC number to load the watershed boundary from a **File Geodatabase** "
        "(`.gdb`) located in `data/huc_gdb/`. This boundary can be used later to clip all layers."
    )

    huc_dir = PROJECT_ROOT / "data" / "huc_gdb"
    huc_number = st.text_input(
        "HUC number (e.g. HUC8 or HUC12)",
        value=st.session_state.get("huc_number_input", ""),
        key="huc_number_input",
        placeholder="e.g. 12030101",
    )

    if huc_number.strip():
        if not huc_dir.exists() or not list(huc_dir.glob("*.gdb")):
            st.warning(
                f"No File Geodatabase (.gdb) found in `{huc_dir}`. "
                "Place a HUC File Geodatabase there to use this feature."
            )
        else:
            if st.button("Load HUC boundary", key="load_huc_btn"):
                try:
                    gdf, meta = find_and_load_huc(huc_dir, huc_number.strip())
                    st.session_state["huc_boundary"] = gdf
                    st.session_state["huc_metadata"] = meta
                    st.success(
                        f"Loaded HUC boundary {meta.get('huc_number', huc_number)} "
                        f"from {meta.get('source')} (layer: {meta.get('layer')})."
                    )
                except Exception as e:
                    st.error(str(e))
                    st.session_state["huc_boundary"] = None
                    st.session_state["huc_metadata"] = None

    if st.session_state.get("huc_boundary") is not None:
        st.info("HUC boundary is loaded and will be used for clipping when enabled in Step 4.")
        if st.button("Clear HUC boundary", key="clear_huc_btn"):
            st.session_state["huc_boundary"] = None
            st.session_state["huc_metadata"] = None
            st.rerun()


def step2_uploads_section() -> None:
    """Step 2: Uploads and Summary — DEM, Land Use, Soil; summary table; map."""
    st.header("Step 2: Upload DEM, Land Use, and Soil Layers")

    st.markdown(
        "Upload one or more files per layer (e.g. multiple DEM tiles). "
        "Rasters: GeoTIFF (`.tif`). Vectors: zipped Shapefile (`.zip`)."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        dem_files = st.file_uploader(
            "DEM (raster)",
            type=["tif", "tiff"],
            accept_multiple_files=True,
            key="dem_uploader",
        )
    with col2:
        landuse_files = st.file_uploader(
            "Land Use (raster or vector)",
            type=["tif", "tiff", "zip"],
            accept_multiple_files=True,
            key="landuse_uploader",
        )
    with col3:
        soil_files = st.file_uploader(
            "Soil Map (raster or vector)",
            type=["tif", "tiff", "zip"],
            accept_multiple_files=True,
            key="soil_uploader",
        )

    # Persist uploads to disk and session state when user provides new files
    if dem_files:
        st.session_state["dem_uploads"] = _save_uploaded_files(
            dem_files, "dem", (".tif", ".tiff")
        )
    if landuse_files:
        st.session_state["landuse_uploads"] = _save_uploaded_files(
            landuse_files, "landuse", (".tif", ".tiff", ".zip")
        )
    if soil_files:
        st.session_state["soil_uploads"] = _save_uploaded_files(
            soil_files, "soil", (".tif", ".tiff", ".zip")
        )

    # Summary table: Layer Name, Type, CRS, Resolution
    st.subheader("Upload summary")
    rows = []
    for label, key in [("DEM", "dem_uploads"), ("Land Use", "landuse_uploads"), ("Soil", "soil_uploads")]:
        uploads = st.session_state.get(key) or []
        for u in uploads:
            meta = u["metadata"]
            rows.append({
                "Layer Name": u["name"],
                "Category": label,
                "Type": meta.get("type", "—"),
                "CRS": meta.get("crs") or "—",
                "Resolution": meta.get("resolution") or "—",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No layers uploaded yet.")

    # Map preview: all layers + HUC
    st.subheader("Map preview (all layers)")
    layers_for_map = []
    for label, key in [("DEM", "dem_uploads"), ("Land Use", "landuse_uploads"), ("Soil", "soil_uploads")]:
        uploads = st.session_state.get(key) or []
        for u in uploads:
            data, meta = _get_layer_data_for_map(u)
            layers_for_map.append((data, meta, f"{label}: {u['name']}"))

    if layers_for_map:
        m = create_multi_layer_map(
            layers_for_map,
            huc_gdf=st.session_state.get("huc_boundary"),
        )
        # Render the underlying folium map directly to HTML to avoid
        # temporary-file locking issues on Windows in leafmap.to_html().
        components.html(m.get_root().render(), height=500)
    else:
        st.info("Upload layers above to see the map.")


def step3_landuse_extraction_section() -> None:
    """Step 3: Land Use class extraction, table, and CSV export with SWAT_Class_Target."""
    st.header("Step 3: Land Use Class Extraction & Export")

    landuse_uploads = st.session_state.get("landuse_uploads") or []
    if not landuse_uploads:
        st.info("Upload at least one Land Use layer in Step 2 first.")
        return

    # Use first land use layer for extraction
    first_lu = landuse_uploads[0]
    meta = first_lu["metadata"]

    if st.button("Extract land use classes", key="extract_lu_btn"):
        try:
            if meta["type"] == "raster":
                df = extract_landuse_classes_raster(first_lu["path"])
            else:
                gdf, _ = load_vector_from_path(Path(first_lu["path"]))
                df = extract_landuse_classes_vector(gdf)
            df["SWAT_Class_Target"] = ""
            st.session_state["land_use_classes_df"] = df
            st.success("Classes extracted.")
        except Exception as e:
            st.error(str(e))

    lu_df = st.session_state.get("land_use_classes_df")
    if lu_df is not None:
        st.subheader("Land use classes (fill SWAT_Class_Target and re-upload in Step 4)")
        st.dataframe(lu_df, use_container_width=True, hide_index=True)

        csv = lu_df.to_csv(index=False)
        st.download_button(
            "Download land use lookup CSV (with blank SWAT_Class_Target)",
            data=csv,
            file_name="land_use_lookup.csv",
            mime="text/csv",
            key="download_lu_lookup",
        )


def step4_preprocessing_options_section() -> None:
    """Step 4: CRS, mosaic, clip to HUC, Land Use reclassification CSV upload."""
    st.header("Step 4: Preprocessing Options")

    # Collect CRS options: common EPSG + from uploaded layers
    crs_options = [
        "EPSG:4326",
        "EPSG:32615",
        "EPSG:32616",
        "EPSG:32617",
        "EPSG:32618",
        "EPSG:26915",
        "EPSG:26916",
        "EPSG:26917",
    ]
    for key in ("dem_uploads", "landuse_uploads", "soil_uploads"):
        for u in st.session_state.get(key) or []:
            c = u["metadata"].get("crs")
            if c and c not in crs_options:
                crs_options.append(c)

    target_crs = st.selectbox(
        "Target CRS (for all layers)",
        options=crs_options,
        index=0,
        key="target_crs_select",
    )
    st.session_state["target_crs"] = target_crs

    do_mosaic = st.checkbox(
        "Mosaic/combine multiple files per layer into one",
        value=st.session_state.get("do_mosaic", False),
        key="do_mosaic_cb",
    )
    st.session_state["do_mosaic"] = do_mosaic

    clip_to_huc = st.checkbox(
        "Clip all layers to HUC watershed (from Step 1)",
        value=st.session_state.get("clip_to_huc", False),
        key="clip_to_huc_cb",
    )
    st.session_state["clip_to_huc"] = clip_to_huc
    if clip_to_huc and st.session_state.get("huc_boundary") is None:
        st.warning("Load a HUC boundary in Step 1 for clipping.")

    st.subheader("Land Use reclassification")
    reclass_csv = st.file_uploader(
        "Upload modified Land Use CSV (with SWAT_Class_Target filled)",
        type=["csv"],
        key="reclass_csv_uploader",
    )
    if reclass_csv:
        try:
            df = pd.read_csv(reclass_csv)
            if "SWAT_Class_Target" in df.columns:
                st.session_state["reclass_lookup_df"] = df
                st.success("Reclassification lookup loaded.")
            else:
                st.error("CSV must contain a column named SWAT_Class_Target.")
        except Exception as e:
            st.error(str(e))


def _run_preprocessing() -> Dict[str, str]:
    """
    Run the preprocessing pipeline: reproject, mosaic, clip, reclassify.
    Returns dict with paths: dem, landuse, soil, lookup_csv.
    """
    cache = Path(st.session_state["upload_cache_dir"])
    out_dir = cache / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_crs = st.session_state["target_crs"]
    do_mosaic = st.session_state["do_mosaic"]
    clip_to_huc = st.session_state["clip_to_huc"]
    huc_gdf = st.session_state.get("huc_boundary")
    reclass_df = st.session_state.get("reclass_lookup_df")

    results = {}

    def _process_raster_list(
        uploads: List[Dict],
        layer_name: str,
        out_name: str,
    ) -> Optional[str]:
        if not uploads:
            return None
        paths = [u["path"] for u in uploads if u["metadata"]["type"] == "raster"]
        if not paths:
            # Vector-only: take first and reproject/clip
            u = uploads[0]
            gdf, _ = load_vector_from_path(Path(u["path"]))
            if target_crs and str(gdf.crs) != target_crs:
                gdf = reproject_vector(gdf, target_crs)
            if clip_to_huc and huc_gdf is not None:
                boundary = huc_gdf.to_crs(gdf.crs) if huc_gdf.crs != gdf.crs else huc_gdf
                gdf = clip_vector_to_geometry(gdf, boundary)
            out_path = out_dir / f"{out_name}.shp"
            gdf.to_file(out_path)
            return str(out_path)
        if do_mosaic and len(paths) > 1:
            merged = out_dir / f"{out_name}_merged.tif"
            mosaic_rasters(paths, merged)
            current = str(merged)
        else:
            current = paths[0]
        if target_crs:
            reproj = out_dir / f"{out_name}_reproj.tif"
            reproject_raster(current, target_crs, reproj)
            current = str(reproj)
        if clip_to_huc and huc_gdf is not None:
            clip_path = out_dir / f"{out_name}_clip.tif"
            clip_raster_to_geometry(current, huc_gdf, clip_path)
            current = str(clip_path)
        return current

    # DEM
    dem_path = _process_raster_list(
        st.session_state.get("dem_uploads") or [],
        "DEM",
        "dem",
    )
    if dem_path:
        results["dem"] = dem_path

    # Soil
    soil_path = _process_raster_list(
        st.session_state.get("soil_uploads") or [],
        "Soil",
        "soil",
    )
    if soil_path:
        results["soil"] = soil_path

    # Land Use: possibly reclassify
    lu_uploads = st.session_state.get("landuse_uploads") or []
    lu_path = _process_raster_list(lu_uploads, "Land Use", "landuse")
    if lu_path and reclass_df is not None and "Value" in reclass_df.columns and "SWAT_Class_Target" in reclass_df.columns:
        reclass_path = out_dir / "landuse_reclassified.tif"
        reclassify_raster_from_lookup(
            lu_path,
            reclass_df,
            value_column="Value",
            target_column="SWAT_Class_Target",
            out_path=reclass_path,
        )
        results["landuse"] = str(reclass_path)
    elif lu_path:
        results["landuse"] = lu_path

    # Lookup CSV
    if st.session_state.get("land_use_classes_df") is not None:
        lookup_path = out_dir / "land_use_lookup.csv"
        st.session_state["land_use_classes_df"].to_csv(lookup_path, index=False)
        results["lookup_csv"] = str(lookup_path)
    elif reclass_df is not None:
        lookup_path = out_dir / "land_use_lookup.csv"
        reclass_df.to_csv(lookup_path, index=False)
        results["lookup_csv"] = str(lookup_path)

    return results


def step5_final_preview_section() -> None:
    """Step 5: Run Preprocessing, final map, download ZIP."""
    st.header("Step 5: Final Preview and Export")

    if st.button("Run Preprocessing", type="primary", key="run_preprocessing_btn"):
        with st.spinner("Running preprocessing…"):
            try:
                outputs = _run_preprocessing()
                st.session_state["processed_outputs"] = outputs
                st.success("Preprocessing finished.")
            except Exception as e:
                st.exception(e)
                st.session_state["processed_outputs"] = None

    outputs = st.session_state.get("processed_outputs")
    if outputs:
        st.subheader("Processed layers")
        layers_for_map = []
        huc_gdf = st.session_state.get("huc_boundary")
        for key, path in outputs.items():
            if key == "lookup_csv":
                continue
            path = Path(path)
            if path.suffix.lower() in (".tif", ".tiff"):
                with rasterio.open(path) as src:
                    meta = {
                        "type": "raster",
                        "bounds": {
                            "left": src.bounds.left, "bottom": src.bounds.bottom,
                            "right": src.bounds.right, "top": src.bounds.top,
                        },
                    }
                layers_for_map.append((str(path), meta, key.upper()))
            else:
                gdf = gpd.read_file(path)
                meta = {"type": "vector", "bounds": {"left": gdf.total_bounds[0], "bottom": gdf.total_bounds[1], "right": gdf.total_bounds[2], "top": gdf.total_bounds[3]}}
                layers_for_map.append((gdf, meta, key.upper()))
        if layers_for_map:
            m = create_multi_layer_map(layers_for_map, huc_gdf=huc_gdf)
            components.html(m.get_root().render(), height=500)

        # Build ZIP for download (include all shapefile components when path is .shp)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for key, path in outputs.items():
                path = Path(path)
                if not path.exists():
                    continue
                if path.suffix.lower() == ".shp":
                    for sibling in path.parent.glob(path.stem + ".*"):
                        zf.write(sibling, sibling.name)
                else:
                    zf.write(path, path.name)
        zip_buffer.seek(0)
        st.download_button(
            "Download processed layers (ZIP)",
            data=zip_buffer,
            file_name="qswat_processed_inputs.zip",
            mime="application/zip",
            key="download_zip_btn",
        )
    else:
        st.info("Click **Run Preprocessing** to run the pipeline and see the final map and download.")


def main() -> None:
    init_page_config()
    init_session_state()

    st.title("QSWAT+ Inputs Preprocessor")
    st.markdown(
        "Follow the steps below: select an optional HUC, upload DEM/Land Use/Soil, "
        "extract land use classes, set options, then run preprocessing and download."
    )

    step1_huc_section()
    st.markdown("---")
    step2_uploads_section()
    st.markdown("---")
    step3_landuse_extraction_section()
    st.markdown("---")
    step4_preprocessing_options_section()
    st.markdown("---")
    step5_final_preview_section()


if __name__ == "__main__":
    main()
