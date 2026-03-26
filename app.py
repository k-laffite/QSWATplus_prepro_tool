"""
QSWAT+ Inputs Preprocessor — Streamlit app.

Sequential workflow:
  1. HUC Watershed Selection
  2. Uploads and Summary (DEM, Land Use, Soil)
  3. Land Use Class Extraction & Export
  4. Advanced Agricultural HRU Definition (optional field-specific soils)
  5. Preprocessing Options (CRS, mosaic, clip, reclassification CSV)
  6. Final Preview and Export
"""

import io
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import rasterio
import streamlit as st
import streamlit.components.v1 as components

from utils.file_handlers import (
    get_numeric_columns,
    get_text_columns,
    load_raster,
    load_vector_from_path,
    load_vector_from_zip,
)
from utils.map_utils import create_multi_layer_map
from utils.spatial_processing import (
    apply_field_specific_soil_overlay,
    ensure_huc_gdb_dir,
    find_and_load_huc,
    get_raster_resolution,
    mosaic_rasters,
    rasterize_vector_to_raster,
    clip_raster_to_geometry,
    clip_vector_to_geometry,
    reproject_raster,
    reproject_vector,
    extract_landuse_classes_raster,
    extract_landuse_classes_vector,
    reclassify_raster_from_lookup,
    warp_raster_to_template,
)

# Project root (directory containing app.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to SWAT+ land use class codes (used in Step 3 dropdown)
SWAT_CLASSES_CSV = PROJECT_ROOT / "data" / "swat+_classes" / "swat+_classes.csv"

# SSURGO / usersoil reference (loaded silently for Step 4 field-specific soils)
SSURGO_SOILS_CSV = PROJECT_ROOT / "data" / "ssurgo_soil_classes" / "SSURGO_Soils.csv"


@st.cache_data(ttl=3600)
def _load_swat_ref() -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, str]]:
    """
    Load SWAT+ reference CSV and return (df, name_to_id, name_to_desc).
    name_to_id: code -> numeric id; name_to_desc: code -> description text.
    Returns (empty df, {}, {}) if file missing or invalid.
    """
    if not SWAT_CLASSES_CSV.exists():
        return pd.DataFrame(), {}, {}
    try:
        df = pd.read_csv(SWAT_CLASSES_CSV)
        for c in ("id", "code", "description"):
            if c not in df.columns:
                return pd.DataFrame(), {}, {}
        df["code"] = df["code"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["code"], keep="first")
        name_to_id = df.set_index("code")["id"].astype(int).to_dict()
        name_to_desc = df.set_index("code")["description"].astype(str).to_dict()
        return df, name_to_id, name_to_desc
    except Exception:
        return pd.DataFrame(), {}, {}


@st.cache_data(ttl=3600)
def _load_ssurgo_soils() -> pd.DataFrame:
    """Load SSURGO usersoil table from bundled CSV (empty DataFrame if missing)."""
    if not SSURGO_SOILS_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(SSURGO_SOILS_CSV)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _load_swat_class_options() -> List[str]:
    """Load valid SWAT+ class codes from data/swat+_classes/swat+_classes.csv. Returns list of 'code' values. Cached."""
    swat_ref_df, name_to_id, _ = _load_swat_ref()
    if swat_ref_df.empty:
        return []
    return swat_ref_df["code"].tolist()


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
        "lu_desc_mapping": None,
        "edited_lu_table": None,
        "reclass_lookup_df": None,
        "target_crs": "EPSG:4326",
        "do_mosaic": False,
        "clip_to_huc": False,
        "reclassify_landuse_to_swat_id": True,
        "processed_outputs": None,
        "upload_cache_dir": None,
        "field_specific_soil_path": None,
        "field_soil_lookup_csv": None,
        "field_soil_usersoil_csv": None,
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
    For DEM: .tif/.tiff only. For Land Use/Soil: .tif, .tiff, .zip, and shapefile components (.shp, .shx, .dbf, .prj).
    Shapefile components with the same base name are grouped into one vector layer.
    """
    if not uploaded_files:
        return []

    cache_dir = Path(st.session_state["upload_cache_dir"])
    layer_dir = cache_dir / layer_key
    layer_dir.mkdir(parents=True, exist_ok=True)
    for f in layer_dir.iterdir():
        try:
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f, ignore_errors=True)
        except OSError:
            pass

    vector_extensions = (".shp", ".shx", ".dbf", ".prj")
    shapefile_groups: Dict[str, List[Any]] = defaultdict(list)
    zips = []
    tifs = []
    for uf in uploaded_files:
        name = uf.name
        suffix = Path(name).suffix.lower()
        if suffix not in accepted_extensions:
            continue
        if suffix == ".zip":
            zips.append(uf)
        elif suffix in (".tif", ".tiff"):
            tifs.append(uf)
        elif suffix in vector_extensions:
            shapefile_groups[Path(name).stem].append(uf)

    results = []
    idx = 0
    for uf in zips:
        name = uf.name
        path = layer_dir / f"{idx}_{name}"
        path.write_bytes(uf.getbuffer())
        try:
            with zipfile.ZipFile(path, "r") as zf:
                extract_dir = layer_dir / f"{idx}_extracted"
                extract_dir.mkdir(exist_ok=True)
                zf.extractall(extract_dir)
            gdf, meta = load_vector_from_path(extract_dir)
            meta["path"] = str(extract_dir)
            meta["name"] = name
            meta["resolution"] = None
            results.append({"path": str(extract_dir), "name": name, "metadata": meta})
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
        idx += 1

    for uf in tifs:
        name = uf.name
        path = layer_dir / f"{idx}_{name}"
        path.write_bytes(uf.getbuffer())
        try:
            _, meta = load_raster(path)
            meta["name"] = name
            results.append({"path": str(path), "name": name, "metadata": meta})
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
        idx += 1

    for stem, group in shapefile_groups.items():
        if not group:
            continue
        # Require at least .shp; .shx and .dbf are typically needed for reading
        has_shp = any(Path(f.name).suffix.lower() == ".shp" for f in group)
        if not has_shp:
            continue
        extract_dir = layer_dir / f"{idx}_shp_{stem}"
        extract_dir.mkdir(exist_ok=True)
        for uf in group:
            (extract_dir / uf.name).write_bytes(uf.getbuffer())
        try:
            gdf, meta = load_vector_from_path(extract_dir)
            meta["path"] = str(extract_dir)
            meta["name"] = group[0].name if len(group) == 1 else f"{stem}.shp (and components)"
            meta["resolution"] = None
            results.append({"path": str(extract_dir), "name": meta["name"], "metadata": meta})
        except Exception as e:
            st.warning(f"Could not load shapefile {stem}: {e}")
        idx += 1

    return results


def _upload_fingerprint(uploaded_files: List[Any]) -> Optional[tuple]:
    """Return a hashable fingerprint of the current upload list so we only replace session state when uploads actually change."""
    if not uploaded_files:
        return None
    return tuple((uf.name, uf.size) for uf in uploaded_files)


@st.cache_data(ttl=3600)
def _cached_load_vector_from_path(_path_str: str) -> Tuple[Any, Dict[str, Any]]:
    """Load vector from path; cached by path string to avoid re-reading on every rerun."""
    return load_vector_from_path(Path(_path_str))


def _get_layer_data_for_map(entry: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Return (data, metadata) for one layer entry for mapping. Uses rasterized_path if set."""
    if entry.get("rasterized_path"):
        return entry["rasterized_path"], entry["raster_metadata"]
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
        st.info("HUC boundary is loaded and will be used for clipping when enabled in Step 5.")
        if st.button("Clear HUC boundary", key="clear_huc_btn"):
            st.session_state["huc_boundary"] = None
            st.session_state["huc_metadata"] = None
            st.rerun()


def step2_uploads_section() -> None:
    """Step 2: Uploads and Summary — DEM, Land Use, Soil; summary table; map."""
    st.header("Step 2: Upload DEM, Land Use, and Soil Layers")

    st.markdown(
        "Upload one or more files per layer (e.g. multiple DEM tiles). "
        "**Rasters:** GeoTIFF (`.tif`). **Vectors:** zipped Shapefile (`.zip`) or components (`.shp`, `.shx`, `.dbf`, `.prj`). "
        "Vector Land Use/Soil layers can be rasterized to match the DEM grid below."
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

    # Persist uploads only when the set of files actually changed (preserves rasterized_path etc.)
    dem_fp = _upload_fingerprint(dem_files)
    if dem_fp is not None and st.session_state.get("last_dem_fingerprint") != dem_fp:
        st.session_state["dem_uploads"] = _save_uploaded_files(
            dem_files, "dem", (".tif", ".tiff")
        )
        st.session_state["last_dem_fingerprint"] = dem_fp
    if dem_fp is None:
        st.session_state["last_dem_fingerprint"] = None

    landuse_fp = _upload_fingerprint(landuse_files)
    if landuse_fp is not None and st.session_state.get("last_landuse_fingerprint") != landuse_fp:
        st.session_state["landuse_uploads"] = _save_uploaded_files(
            landuse_files, "landuse",
            (".tif", ".tiff", ".zip", ".shp", ".shx", ".dbf", ".prj"),
        )
        st.session_state["last_landuse_fingerprint"] = landuse_fp
    if landuse_fp is None:
        st.session_state["last_landuse_fingerprint"] = None

    soil_fp = _upload_fingerprint(soil_files)
    if soil_fp is not None and st.session_state.get("last_soil_fingerprint") != soil_fp:
        st.session_state["soil_uploads"] = _save_uploaded_files(
            soil_files, "soil",
            (".tif", ".tiff", ".zip", ".shp", ".shx", ".dbf", ".prj"),
        )
        st.session_state["last_soil_fingerprint"] = soil_fp
    if soil_fp is None:
        st.session_state["last_soil_fingerprint"] = None

    # Vector attribute selection and rasterization for Land Use / Soil
    cache_dir = Path(st.session_state["upload_cache_dir"])
    dem_uploads = st.session_state.get("dem_uploads") or []
    template_dem_path = dem_uploads[0]["path"] if dem_uploads else None

    for label, layer_key in [("Land Use", "landuse_uploads"), ("Soil", "soil_uploads")]:
        uploads = st.session_state.get(layer_key) or []
        for i, u in enumerate(uploads):
            meta = u.get("metadata", {})
            if meta.get("type") != "vector":
                continue
            gdf, _ = _cached_load_vector_from_path(u["path"])
            numeric_cols = get_numeric_columns(gdf)
            if not numeric_cols:
                st.warning(f"**{label}** layer «{u['name']}» has no numeric columns; cannot rasterize.")
                continue
            default_ix = numeric_cols.index("GRIDCODE") if "GRIDCODE" in numeric_cols else 0
            col = st.selectbox(
                f"Select the numeric column to use for raster values — {label}: {u['name']}",
                options=numeric_cols,
                index=default_ix,
                key=f"vec_col_{layer_key}_{i}",
            )

            text_cols = get_text_columns(gdf)
            desc_options = ["None"] + text_cols
            desc_col = st.selectbox(
                "Select the column containing Land Use descriptions (optional)",
                options=desc_options,
                index=0,
                key=f"vec_desc_col_{layer_key}_{i}",
            )

            res_method = st.radio(
                "Select Resolution Method",
                options=["Match uploaded DEM", "Provide custom resolution"],
                key=f"vec_res_method_{layer_key}_{i}",
                horizontal=True,
            )

            template_path = None
            target_res = None
            can_rasterize = False

            if res_method == "Match uploaded DEM":
                if template_dem_path:
                    template_path = template_dem_path
                    can_rasterize = True
                else:
                    st.warning(
                        "Please upload a DEM first, or choose a custom resolution."
                    )
                    can_rasterize = False
            else:
                target_res = st.number_input(
                    "Cell Size (in map units)",
                    min_value=0.1,
                    value=30.0,
                    step=1.0,
                    key=f"vec_res_{layer_key}_{i}",
                    help="Resolution used to build the output grid from the shapefile extent.",
                )
                can_rasterize = True

            convert_clicked = st.button(
                "Convert to Raster",
                key=f"convert_btn_{layer_key}_{i}",
                disabled=not (col and can_rasterize and (template_path or target_res is not None)),
            )
            if convert_clicked and col and can_rasterize and (template_path or target_res is not None):
                layer_dir = cache_dir / layer_key
                layer_dir.mkdir(parents=True, exist_ok=True)
                out_path = layer_dir / f"rasterized_{i}.tif"
                try:
                    rasterize_vector_to_raster(
                        gdf,
                        col,
                        out_path,
                        template_raster_path=template_path,
                        target_resolution=target_res if not template_path else None,
                    )
                    _, rmeta = load_raster(out_path)
                    rmeta["name"] = u["name"]
                    u["rasterized_path"] = str(out_path)
                    u["raster_metadata"] = rmeta
                    if label == "Land Use" and desc_col and desc_col != "None":
                        mapping = {}
                        for val in gdf[col].drop_duplicates():
                            try:
                                v = int(val) if pd.notna(val) else None
                                if v is not None:
                                    desc = gdf.loc[gdf[col] == val, desc_col].iloc[0]
                                    mapping[v] = str(desc) if pd.notna(desc) else ""
                            except (ValueError, TypeError):
                                pass
                        st.session_state["lu_desc_mapping"] = mapping
                    st.success(f"Rasterized {u['name']}.")
                except Exception as ex:
                    st.error(f"Rasterization failed for {u['name']}: {ex}")

    # Summary table: Layer Name, Type, CRS, Resolution
    st.subheader("Upload summary")
    rows = []
    for label, key in [("DEM", "dem_uploads"), ("Land Use", "landuse_uploads"), ("Soil", "soil_uploads")]:
        uploads = st.session_state.get(key) or []
        for u in uploads:
            meta = u.get("raster_metadata") or u.get("metadata")
            layer_type = meta.get("type", "—")
            rows.append({
                "Layer Name": u["name"],
                "Category": label,
                "Type": layer_type,
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

    # Use first land use layer for extraction (rasterized or native)
    first_lu = landuse_uploads[0]
    meta = first_lu.get("raster_metadata") or first_lu["metadata"]

    if st.button("Extract land use classes", key="extract_lu_btn"):
        try:
            if first_lu.get("rasterized_path") or meta["type"] == "raster":
                path = first_lu.get("rasterized_path") or first_lu["path"]
                df = extract_landuse_classes_raster(path)
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
        # Use edited table if available so user's SWAT_Class_Target selections persist
        prev_edited = st.session_state.get("edited_lu_table")
        if prev_edited is not None and "Value" in prev_edited.columns and "SWAT_Class_Target" in prev_edited.columns:
            display_df = prev_edited.copy()
        else:
            display_df = lu_df.copy()

        lu_desc_mapping = st.session_state.get("lu_desc_mapping")
        if "Description" not in display_df.columns:
            if lu_desc_mapping is not None:
                desc_vals = display_df["Value"].apply(
                    lambda x: lu_desc_mapping.get(int(x), "") if pd.notna(x) else ""
                )
                display_df.insert(1, "Description", desc_vals)
            else:
                display_df.insert(1, "Description", "")

        # Ensure a stable row index so edits don't cause rows to jump/reset
        if "Value" in display_df.columns:
            display_df = display_df.set_index("Value", drop=False)

        # Load SWAT+ reference and build code -> id, code -> description mappings
        swat_ref_df, name_to_id, name_to_desc = _load_swat_ref()
        if "SWAT_Class_Target" in display_df.columns:
            swat_target = display_df["SWAT_Class_Target"].astype(str).str.strip()
            swat_target = swat_target.replace("", pd.NA).replace("nan", pd.NA)
            id_vals = swat_target.map(name_to_id)
            display_df["SWAT_ID"] = id_vals.apply(lambda x: "" if pd.isna(x) else str(int(x)))
            desc_vals = swat_target.map(name_to_desc)
            display_df["SWAT_Description"] = desc_vals.fillna("").astype(str).replace("nan", "")
        else:
            display_df["SWAT_ID"] = ""
            display_df["SWAT_Description"] = ""

        st.subheader("Land use classes (edit SWAT_Class_Target below)")
        swat_class_options = _load_swat_class_options()
        select_options = [""] + swat_class_options if swat_class_options else []
        if not swat_class_options:
            st.warning(
                "SWAT+ class codes not found (expected data/swat+_classes/swat+_classes.csv). "
                "Using free-text for SWAT_Class_Target."
            )
        if select_options:
            swat_col_config = st.column_config.SelectboxColumn(
                "SWAT_Class_Target",
                options=select_options,
                help="Select the corresponding SWAT+ land use code from the dropdown.",
                required=False,
            )
        else:
            swat_col_config = st.column_config.TextColumn("SWAT_Class_Target", disabled=False)
        column_config = {
            "Value": st.column_config.NumberColumn("Value", disabled=True),
            "Description": st.column_config.TextColumn("Description", disabled=True),
            "Count": st.column_config.NumberColumn("Count", disabled=True),
            "SWAT_Class_Target": swat_col_config,
            "SWAT_ID": st.column_config.TextColumn("SWAT_ID", disabled=True, help="SWAT+ class ID (auto-filled from selection)."),
            "SWAT_Description": st.column_config.TextColumn("SWAT_Description", disabled=True, help="SWAT+ class description (auto-filled from selection)."),
        }
        edited = st.data_editor(
            display_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            key="lu_table_editor",
        )
        st.session_state["edited_lu_table"] = edited

        csv = edited.to_csv(index=False)
        st.download_button(
            "Download land use lookup CSV",
            data=csv,
            file_name="land_use_lookup.csv",
            mime="text/csv",
            key="download_lu_lookup",
        )


def step4_advanced_agricultural_hru_section() -> None:
    """Step 4: Optional field-specific soil HRU raster and usersoil / lookup CSVs."""
    st.header("Step 4: Advanced Agricultural HRU Definition")

    _ = _load_ssurgo_soils()

    with st.expander("Field-specific soils (optional)", expanded=False):
        st.markdown(
            "Upload **crop field** polygons. The app aligns the soil layer to your **first DEM** grid, "
            "rasterizes fields with a unique **Field_ID**, then assigns new raster IDs for each "
            "**(Field_ID, MUKEY)** combination inside fields. Background pixels keep original MUKEY values. "
            "Requires **SSURGO_Soils.csv** at `data/ssurgo_soil_classes/SSURGO_Soils.csv` (loaded automatically)."
        )

        if not SSURGO_SOILS_CSV.exists():
            st.warning(
                f"SSURGO reference not found: `{SSURGO_SOILS_CSV}`. "
                "Add this file to enable field-specific usersoil generation."
            )

        crop_files = st.file_uploader(
            "Crop fields (shapefile .zip or .shp + components)",
            type=["zip", "shp", "shx", "dbf", "prj"],
            accept_multiple_files=True,
            key="crop_fields_uploader",
        )

        c1, c2 = st.columns(2)
        with c1:
            process_fields = st.button("Process Field-Specific Soils", key="process_field_soils_btn")
        with c2:
            if st.button("Clear field-specific soil outputs", key="clear_field_soils_btn"):
                st.session_state["field_specific_soil_path"] = None
                st.session_state["field_soil_lookup_csv"] = None
                st.session_state["field_soil_usersoil_csv"] = None
                st.success("Field-specific soil outputs cleared; preprocessing will use the standard soil layer.")
                st.rerun()

        if process_fields:
            ssurgo_df = _load_ssurgo_soils()
            if ssurgo_df.empty:
                st.error("SSURGO_Soils.csv is missing or could not be read. Cannot build usersoil / lookup tables.")
            elif not crop_files:
                st.error("Upload a crop fields shapefile (.zip or components) first.")
            else:
                dem_uploads = st.session_state.get("dem_uploads") or []
                soil_uploads = st.session_state.get("soil_uploads") or []
                if not dem_uploads:
                    st.error("Upload a DEM in Step 2 first (soil is aligned to the DEM grid).")
                elif not soil_uploads:
                    st.error("Upload a soil layer in Step 2 first.")
                else:
                    cache = Path(st.session_state["upload_cache_dir"])
                    out_dir = cache / "field_specific"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    dem_path = dem_uploads[0]["path"]

                    try:
                        saved_cf = _save_uploaded_files(
                            list(crop_files),
                            "crop_fields",
                            (".zip", ".shp", ".shx", ".dbf", ".prj"),
                        )
                        if not saved_cf:
                            st.error("Could not save crop field uploads.")
                        else:
                            fields_gdf, _ = load_vector_from_path(Path(saved_cf[0]["path"]))
                            soil_u = soil_uploads[0]
                            soil_aligned_path = out_dir / "soil_on_dem.tif"

                            if soil_u.get("metadata", {}).get("type") == "vector":
                                gdf_soil, _ = load_vector_from_path(Path(soil_u["path"]))
                                numeric_cols = get_numeric_columns(gdf_soil)
                                if not numeric_cols:
                                    st.error("Soil vector has no numeric columns; cannot rasterize MUKEY values.")
                                else:
                                    value_col = "MUKEY" if "MUKEY" in numeric_cols else numeric_cols[0]
                                    rasterize_vector_to_raster(
                                        gdf_soil,
                                        value_col,
                                        soil_aligned_path,
                                        template_raster_path=dem_path,
                                    )
                            else:
                                src_soil = soil_u.get("rasterized_path") or soil_u["path"]
                                warp_raster_to_template(src_soil, dem_path, soil_aligned_path)

                            out_soil = out_dir / "soil_field_specific.tif"
                            out_lookup = out_dir / "lookup_soil.csv"
                            out_usersoil = out_dir / "usersoil.csv"

                            info = apply_field_specific_soil_overlay(
                                str(soil_aligned_path),
                                fields_gdf,
                                ssurgo_df,
                                out_soil,
                                out_lookup,
                                out_usersoil,
                            )

                            st.session_state["field_specific_soil_path"] = str(out_soil)
                            st.session_state["field_soil_lookup_csv"] = str(out_lookup)
                            st.session_state["field_soil_usersoil_csv"] = str(out_usersoil)

                            note = (info.get("usersoil_note") or "").strip()
                            msg = (
                                f"Field-specific soil raster saved "
                                f"({info.get('n_new_combos', 0)} new field–soil combinations)."
                            )
                            if note:
                                msg += f" {note}"
                            st.success(msg)
                    except Exception as e:
                        st.exception(e)


def step5_preprocessing_options_section() -> None:
    """Step 5: CRS, mosaic, clip to HUC, Land Use reclassification CSV upload."""
    st.header("Step 5: Preprocessing Options")

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
            c = (u.get("raster_metadata") or u.get("metadata") or {}).get("crs")
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

    reclassify_to_swat_id = st.checkbox(
        "Reclassify Land Use raster to SWAT_ID values",
        value=st.session_state.get("reclassify_landuse_to_swat_id", True),
        key="reclassify_landuse_to_swat_id_cb",
    )
    st.session_state["reclassify_landuse_to_swat_id"] = reclassify_to_swat_id


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
    reclassify_to_swat_id = st.session_state.get("reclassify_landuse_to_swat_id", True)
    huc_gdf = st.session_state.get("huc_boundary")
    edited_lu = st.session_state.get("edited_lu_table")
    if edited_lu is not None:
        reclass_df = edited_lu
    else:
        reclass_df = st.session_state.get("reclass_lookup_df")

    results = {}

    def _process_raster_list(
        uploads: List[Dict],
        layer_name: str,
        out_name: str,
    ) -> Optional[str]:
        if not uploads:
            return None
        # Include native rasters and rasterized vector layers
        paths = []
        for u in uploads:
            if u.get("rasterized_path"):
                paths.append(u["rasterized_path"])
            elif u.get("metadata", {}).get("type") == "raster":
                paths.append(u["path"])
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

    # Soil (optional override from Step 4 field-specific processing)
    field_soil = st.session_state.get("field_specific_soil_path")
    if field_soil and Path(field_soil).exists():
        fake_soil_uploads = [
            {
                "path": field_soil,
                "name": "soil_field_specific.tif",
                "metadata": {"type": "raster"},
                "rasterized_path": None,
            }
        ]
        soil_path = _process_raster_list(fake_soil_uploads, "Soil", "soil")
    else:
        soil_path = _process_raster_list(
            st.session_state.get("soil_uploads") or [],
            "Soil",
            "soil",
        )
    if soil_path:
        results["soil"] = soil_path

    # Land Use: optionally reclassify to SWAT_ID values
    lu_uploads = st.session_state.get("landuse_uploads") or []
    lu_path = _process_raster_list(lu_uploads, "Land Use", "landuse")

    if lu_path and reclassify_to_swat_id and reclass_df is not None and "SWAT_ID" in reclass_df.columns:
        # Filter to rows where SWAT_ID was successfully assigned (non-empty)
        swat_id_str = reclass_df["SWAT_ID"].astype(str).str.strip()
        has_swat_id = (swat_id_str != "") & (swat_id_str != "nan")
        mapped_df = reclass_df.loc[has_swat_id, ["Value", "SWAT_ID"]].dropna(subset=["SWAT_ID"])
        if not mapped_df.empty:
            reclass_path = out_dir / "landuse_reclassified.tif"
            reclassify_raster_from_lookup(
                lu_path,
                mapped_df,
                value_column="Value",
                target_column="SWAT_ID",
                out_path=reclass_path,
            )
            results["landuse"] = str(reclass_path)

            # Clean lookup: unique SWAT_ID, SWAT_Class_Target, SWAT_Description (drop duplicates)
            lookup_cols = [c for c in ("SWAT_ID", "SWAT_Class_Target", "SWAT_Description") if c in reclass_df.columns]
            if lookup_cols:
                clean_lookup = reclass_df.loc[has_swat_id, lookup_cols].drop_duplicates()
                swat_lookup_path = out_dir / "swat_landuse_lookup.csv"
                clean_lookup.to_csv(swat_lookup_path, index=False)
                results["swat_landuse_lookup_csv"] = str(swat_lookup_path)
        else:
            results["landuse"] = lu_path
    elif lu_path:
        results["landuse"] = lu_path

    # Full lookup CSV (always export when we have a table)
    if reclass_df is not None:
        lookup_path = out_dir / "land_use_lookup.csv"
        reclass_df.to_csv(lookup_path, index=False)
        results["lookup_csv"] = str(lookup_path)
    elif st.session_state.get("land_use_classes_df") is not None:
        lookup_path = out_dir / "land_use_lookup.csv"
        st.session_state["land_use_classes_df"].to_csv(lookup_path, index=False)
        results["lookup_csv"] = str(lookup_path)

    lk_soil = st.session_state.get("field_soil_lookup_csv")
    if lk_soil and Path(lk_soil).exists():
        dst = out_dir / "lookup_soil.csv"
        shutil.copy(lk_soil, dst)
        results["lookup_soil_csv"] = str(dst)

    us_soil = st.session_state.get("field_soil_usersoil_csv")
    if us_soil and Path(us_soil).exists():
        dst_u = out_dir / "usersoil.csv"
        shutil.copy(us_soil, dst_u)
        results["usersoil_csv"] = str(dst_u)

    return results


def step6_final_preview_section() -> None:
    """Step 6: Run Preprocessing, final map, download ZIP."""
    st.header("Step 6: Final Preview and Export")

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
            if key in (
                "lookup_csv",
                "swat_landuse_lookup_csv",
                "lookup_soil_csv",
                "usersoil_csv",
            ):
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
        "extract land use classes, optionally define field-specific soils, set options, "
        "then run preprocessing and download."
    )

    step1_huc_section()
    st.markdown("---")
    step2_uploads_section()
    st.markdown("---")
    step3_landuse_extraction_section()
    st.markdown("---")
    step4_advanced_agricultural_hru_section()
    st.markdown("---")
    step5_preprocessing_options_section()
    st.markdown("---")
    step6_final_preview_section()


if __name__ == "__main__":
    main()
