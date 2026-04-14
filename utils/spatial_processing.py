"""
Spatial processing utilities for QSWAT+ preprocessing: HUC lookup,
mosaic, clip, reproject, and land use class extraction/reclassification.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize as rasterio_rasterize
from rasterio.merge import merge as rasterio_merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds, array_bounds
import rasterio.mask

# Allowed HUC attribute column names in the HUC File Geodatabase.
# Per requirements, we ONLY check these exact names.
HUC_COLUMN_CANDIDATES = (
    "huc2",
    "huc4",
    "huc6",
    "huc8",
    "huc10",
    "huc12",
    "huc14",
    "huc16",
)


def ensure_huc_gdb_dir(project_root: Path) -> Path:
    """Create data/huc_gdb/ under project root if it doesn't exist."""
    project_root = Path(project_root)
    huc_dir = project_root / "data" / "huc_gdb"
    huc_dir.mkdir(parents=True, exist_ok=True)
    return huc_dir


def _find_huc_column(gdf: gpd.GeoDataFrame, huc_number: str) -> Optional[str]:
    """Return the first allowed HUC column that contains the given HUC value."""
    for col in HUC_COLUMN_CANDIDATES:
        if col in gdf.columns:
            matches = gdf[col].astype(str).str.strip() == str(huc_number).strip()
            if matches.any():
                return col
    return None


def find_and_load_huc(
    huc_gdb_dir: Path,
    huc_number: str,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Scan huc_dir for shapefiles, find the feature matching huc_number, and return it.

    Parameters
    ----------
    huc_gdb_dir : Path
        Directory containing one or more File Geodatabases (.gdb) with HUC polygons.
    huc_number : str
        HUC code to find (e.g. "12030101" for HUC8).

    Returns
    -------
    (GeoDataFrame, dict)
        Single-feature GeoDataFrame (the HUC polygon) and metadata dict.
    """
    huc_gdb_dir = Path(huc_gdb_dir)
    huc_number = str(huc_number).strip()
    if not huc_number:
        raise ValueError("HUC number cannot be empty.")

    gdb_paths = [p for p in huc_gdb_dir.glob("*.gdb") if p.is_dir()]
    if not gdb_paths:
        raise FileNotFoundError(f"No File Geodatabase (.gdb) directories found in {huc_gdb_dir}")

    for gdb_path in gdb_paths:
        layers = fiona.listlayers(str(gdb_path))
        for layer_name in layers:
            gdf = gpd.read_file(gdb_path, layer=layer_name)
            if gdf.crs is None or gdf.empty:
                continue
            col = _find_huc_column(gdf, huc_number)
            if col is None:
                continue
            subset = gdf[gdf[col].astype(str).str.strip() == huc_number].copy()
            if subset.empty:
                continue
            metadata: Dict[str, Any] = {
                "type": "vector",
                "source": str(gdb_path),
                "layer": layer_name,
                "crs": str(gdf.crs),
                "huc_number": huc_number,
                "huc_column": col,
                "bounds": _bounds_dict(subset.total_bounds),
            }
            return subset, metadata

    raise ValueError(
        f"No feature with HUC number '{huc_number}' found in any .gdb under {huc_gdb_dir}. "
        "Check the HUC value and ensure a HUC File Geodatabase is in data/huc_gdb/."
    )


def _bounds_dict(bounds: Any) -> Dict[str, float]:
    """Return bounds as a serializable dict."""
    left, bottom, right, top = bounds
    return {"left": float(left), "bottom": float(bottom), "right": float(right), "top": float(top)}


def _grid_from_vector_extent(
    gdf: gpd.GeoDataFrame,
    template_raster_path: Optional[str] = None,
    target_resolution: Optional[float] = None,
) -> Tuple[Any, Tuple[int, int], Any]:
    """Build an output grid from vector extent, optionally snapped to a template raster grid."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS set.")

    if template_raster_path:
        with rasterio.open(template_raster_path) as src:
            template_crs = src.crs
            template_transform = src.transform
        if gdf.crs != template_crs:
            gdf = gdf.to_crs(template_crs)

        left, bottom, right, top = gdf.total_bounds
        xres = float(abs(template_transform.a))
        yres = float(abs(template_transform.e))
        x_origin = float(template_transform.c)
        y_origin = float(template_transform.f)

        snapped_left = x_origin + math.floor((left - x_origin) / xres) * xres
        snapped_right = x_origin + math.ceil((right - x_origin) / xres) * xres
        snapped_top = y_origin - math.floor((y_origin - top) / yres) * yres
        snapped_bottom = y_origin - math.ceil((y_origin - bottom) / yres) * yres

        width_px = max(1, int(round((snapped_right - snapped_left) / xres)))
        height_px = max(1, int(round((snapped_top - snapped_bottom) / yres)))

        out_transform = from_bounds(
            snapped_left,
            snapped_bottom,
            snapped_right,
            snapped_top,
            width_px,
            height_px,
        )
        out_shape = (height_px, width_px)
        out_crs = template_crs
    else:
        if target_resolution is None or target_resolution <= 0:
            raise ValueError("target_resolution (e.g. 30 meters) is required when no template raster is provided.")
        left, bottom, right, top = gdf.total_bounds
        width_px = int((right - left) / target_resolution)
        height_px = int((top - bottom) / target_resolution)
        width_px = max(1, width_px)
        height_px = max(1, height_px)
        out_transform = from_bounds(left, bottom, right, top, width_px, height_px)
        out_shape = (height_px, width_px)
        out_crs = gdf.crs

    return out_transform, out_shape, out_crs


def rasterize_vector_to_raster(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    out_path: Path,
    template_raster_path: Optional[str] = None,
    target_resolution: Optional[float] = None,
    nodata: int = -9999,
    output_transform: Optional[Any] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    output_crs: Optional[Any] = None,
) -> str:
    """
    Rasterize a GeoDataFrame using a numeric attribute for cell values.

    If template_raster_path is provided (e.g. a DEM), the output grid uses that
    raster's CRS and cell size, snapped to the DEM grid origin, while expanding
    to cover the full GeoDataFrame extent. This preserves the full vector footprint
    instead of clipping to the DEM footprint. Otherwise, target_resolution (in CRS
    units, e.g. meters) is required and the grid is built from the GeoDataFrame's extent.

    Parameters
    ----------
    gdf : GeoDataFrame
        Vector layer to rasterize.
    value_column : str
        Column name whose values become raster pixel values (must be numeric).
    out_path : Path
        Output GeoTIFF path.
    template_raster_path : str, optional
        Path to a raster (e.g. DEM) to use for CRS and cell size. The output grid
        is snapped to the template grid while covering the full vector extent.
    target_resolution : float, optional
        Resolution in CRS units (e.g. meters) when no template is used.
    nodata : int
        NoData value for the output raster.

    Returns
    -------
    str
        Path to the written raster.
    """
    out_path = Path(out_path)
    gdf = gdf.copy()
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS set.")

    if output_transform is not None and output_shape is not None and output_crs is not None:
        out_transform = output_transform
        out_shape = output_shape
        out_crs = output_crs
        if gdf.crs != out_crs:
            gdf = gdf.to_crs(out_crs)
    else:
        out_transform, out_shape, out_crs = _grid_from_vector_extent(
            gdf,
            template_raster_path=template_raster_path,
            target_resolution=target_resolution,
        )
        if gdf.crs != out_crs:
            gdf = gdf.to_crs(out_crs)

    # Build (geometry, value) pairs; use numeric type for raster
    values = gdf[value_column]
    if not pd.api.types.is_numeric_dtype(values):
        values = pd.to_numeric(values, errors="coerce")
    shapes = [(geom, int(val) if pd.notna(val) else nodata) for geom, val in zip(gdf.geometry, values)]

    rasterized = rasterio_rasterize(
        shapes,
        out_shape=out_shape,
        transform=out_transform,
        fill=nodata,
        dtype="int32",
        nodata=nodata,
    )

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=out_shape[0],
        width=out_shape[1],
        count=1,
        dtype="int32",
        crs=out_crs,
        transform=out_transform,
        nodata=nodata,
    ) as dest:
        dest.write(rasterized, 1)

    return str(out_path)


def combine_aligned_rasters(
    raster_paths: List[str],
    out_path: Path,
    nodata: int = -9999,
) -> str:
    """Combine aligned categorical rasters by taking the first non-nodata pixel."""
    if not raster_paths:
        raise ValueError("At least one raster path is required.")

    out_path = Path(out_path)
    with rasterio.open(raster_paths[0]) as src0:
        base = src0.read(1)
        meta = src0.meta.copy()

    out_arr = base.copy()
    for p in raster_paths[1:]:
        with rasterio.open(p) as src:
            arr = src.read(1)
        fill_mask = (out_arr == nodata) & (arr != nodata)
        out_arr[fill_mask] = arr[fill_mask]

    meta.update({"nodata": nodata})
    with rasterio.open(out_path, "w", **meta) as dest:
        dest.write(out_arr, 1)
    return str(out_path)


def get_raster_resolution(path: str) -> Optional[str]:
    """Return a short resolution string (e.g. '30.0 m') from a raster path."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as src:
            res = src.res
            if res and len(res) >= 2:
                return f"{float(res[0]):.4g} m"
            return None
    except Exception:
        return None


def mosaic_rasters(
    paths: List[str],
    out_path: Path,
) -> str:
    """
    Merge multiple raster files into one. All inputs must share CRS and band count.

    Parameters
    ----------
    paths : list of str
        Paths to GeoTIFF (or other raster) files.
    out_path : Path
        Output raster path.

    Returns
    -------
    str
        Path to the merged raster.
    """
    if not paths:
        raise ValueError("At least one raster path is required.")
    out_path = Path(out_path)

    sources = []
    for p in paths:
        src = rasterio.open(p)
        sources.append(src)

    try:
        mosaic_data, mosaic_transform = rasterio_merge(sources, method="first")
        out_meta = sources[0].meta.copy()
        out_meta.update({
            "height": mosaic_data.shape[1],
            "width": mosaic_data.shape[2],
            "transform": mosaic_transform,
        })
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic_data)
    finally:
        for src in sources:
            src.close()

    return str(out_path)


def clip_raster_to_geometry(
    raster_path: str,
    boundary_gdf: gpd.GeoDataFrame,
    out_path: Path,
) -> str:
    """
    Clip a raster to the extent of the boundary geometry (e.g. HUC polygon).

    Parameters
    ----------
    raster_path : str
        Path to input raster.
    boundary_gdf : GeoDataFrame
        Single or multiple polygons; will be converted to raster CRS if needed.
    out_path : Path
        Output clipped raster path.

    Returns
    -------
    str
        Path to the clipped raster.
    """
    out_path = Path(out_path)
    with rasterio.open(raster_path) as src:
        src_crs = src.crs
        if boundary_gdf.crs != src_crs:
            boundary_gdf = boundary_gdf.to_crs(src_crs)
        shapes = [feature for feature in boundary_gdf.geometry]
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, filled=True, nodata=src.nodata if src.nodata is not None else -9999
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": src.nodata if src.nodata is not None else -9999,
        })
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)

    return str(out_path)


def clip_vector_to_geometry(
    gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Clip a vector layer to the boundary. Both must have CRS set; boundary is used as clip mask."""
    if gdf.crs != boundary_gdf.crs:
        boundary_gdf = boundary_gdf.to_crs(gdf.crs)
    return gpd.clip(gdf, boundary_gdf)


def warp_raster_to_template(
    source_path: str,
    template_raster_path: str,
    out_path: Path,
) -> str:
    """
    Resample source raster onto the template raster's grid (transform, shape, CRS).
    Uses nearest-neighbor resampling (appropriate for soil / categorical IDs).
    """
    out_path = Path(out_path)
    with rasterio.open(template_raster_path) as tpl:
        dst_transform = tpl.transform
        dst_crs = tpl.crs
        dst_w = tpl.width
        dst_h = tpl.height
        tpl_meta = tpl.meta.copy()

    with rasterio.open(source_path) as src:
        nodata_out = src.nodata
        if nodata_out is None:
            nodata_out = -9999
        dst = np.full((dst_h, dst_w), nodata_out, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        tpl_meta.update({
            "driver": "GTiff",
            "height": dst_h,
            "width": dst_w,
            "transform": dst_transform,
            "crs": dst_crs,
            "count": 1,
            "dtype": "int32",
            "nodata": int(nodata_out) if np.isfinite(nodata_out) else -9999,
        })
        out_data = np.rint(dst).astype(np.int32)
        with rasterio.open(out_path, "w", **tpl_meta) as dest:
            dest.write(out_data, 1)

    return str(out_path)


def warp_raster_to_template_grid(
    source_path: str,
    template_raster_path: str,
    out_path: Path,
) -> str:
    """
    Resample source raster to match the template raster's CRS and cell size while
    snapping to the template grid origin and preserving the full source extent.
    Uses nearest-neighbor resampling for categorical rasters.
    """
    out_path = Path(out_path)
    with rasterio.open(template_raster_path) as tpl:
        tpl_transform = tpl.transform
        dst_crs = tpl.crs
        xres = float(abs(tpl_transform.a))
        yres = float(abs(tpl_transform.e))
        x_origin = float(tpl_transform.c)
        y_origin = float(tpl_transform.f)

    with rasterio.open(source_path) as src:
        src_bounds = src.bounds
        nodata_out = src.nodata
        if nodata_out is None:
            nodata_out = -9999

        left, bottom, right, top = src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top
        if src.crs != dst_crs:
            dst_bounds = rasterio.warp.transform_bounds(src.crs, dst_crs, left, bottom, right, top)
            left, bottom, right, top = dst_bounds

        snapped_left = x_origin + math.floor((left - x_origin) / xres) * xres
        snapped_right = x_origin + math.ceil((right - x_origin) / xres) * xres
        snapped_top = y_origin - math.floor((y_origin - top) / yres) * yres
        snapped_bottom = y_origin - math.ceil((y_origin - bottom) / yres) * yres

        dst_w = max(1, int(round((snapped_right - snapped_left) / xres)))
        dst_h = max(1, int(round((snapped_top - snapped_bottom) / yres)))
        dst_transform = from_bounds(
            snapped_left,
            snapped_bottom,
            snapped_right,
            snapped_top,
            dst_w,
            dst_h,
        )

        dst = np.full((dst_h, dst_w), nodata_out, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": dst_h,
            "width": dst_w,
            "transform": dst_transform,
            "crs": dst_crs,
            "count": 1,
            "dtype": "int32",
            "nodata": int(nodata_out) if np.isfinite(nodata_out) else -9999,
        })
        out_data = np.rint(dst).astype(np.int32)
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_data, 1)

    return str(out_path)


def reproject_raster(
    raster_path: str,
    target_crs: str,
    out_path: Path,
) -> str:
    """Reproject a raster to target_crs (e.g. 'EPSG:4326') and write to out_path."""
    out_path = Path(out_path)
    target_crs = str(target_crs).strip()
    if not target_crs.startswith("EPSG:"):
        target_crs = f"EPSG:{target_crs}" if target_crs.isdigit() else target_crs

    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
        out_meta = src.meta.copy()
        out_meta.update({"crs": target_crs, "transform": transform, "width": width, "height": height})
        with rasterio.open(out_path, "w", **out_meta) as dest:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dest, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
    return str(out_path)


def reproject_vector(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to target_crs (e.g. 'EPSG:4326')."""
    return gdf.to_crs(target_crs)


def extract_landuse_classes_raster(raster_path: str) -> pd.DataFrame:
    """
    Read a land use raster and return a DataFrame with unique values and pixel counts.
    Column names: Value, Count (and optionally SWAT_Class_Target for export).
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        nodata = src.nodata
    if nodata is not None:
        data = np.ma.masked_equal(data.ravel(), nodata)
    else:
        data = data.ravel()
    values, counts = np.unique(data.compressed() if hasattr(data, "compressed") else data, return_counts=True)
    df = pd.DataFrame({"Value": values.astype(int), "Count": counts})
    df = df.sort_values("Value")
    df["SWAT_Class_Target"] = ""
    return df


def extract_landuse_classes_vector(
    gdf: gpd.GeoDataFrame,
    class_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract unique land use classes from a vector layer and their counts.
    If class_column is None, use the first non-geometry column that looks like a code/class.
    """
    col = class_column
    if not col:
        for c in ("CLASS", "CODE", "VALUE", "LU_CODE", "LANDUSE", "GRIDCODE", "DN"):
            if c in gdf.columns:
                col = c
                break
        if not col:
            non_geom = [c for c in gdf.columns if c != gdf.geometry.name]
            col = non_geom[0] if non_geom else None
    if col is None:
        raise ValueError("Could not determine land use class column. Specify class_column.")
    counts = gdf[col].value_counts().sort_index()
    df = pd.DataFrame({"Value": counts.index.astype(str), "Count": counts.values})
    df["SWAT_Class_Target"] = ""
    return df


def reclassify_raster_from_lookup(
    raster_path: str,
    lookup_df: pd.DataFrame,
    value_column: str,
    target_column: str,
    out_path: Path,
    nodata_out: Optional[int] = None,
) -> str:
    """
    Produce a new raster with pixel values replaced by the target column from the lookup.
    value_column: column in lookup_df that matches raster values (e.g. 'Value').
    target_column: column with new codes (e.g. 'SWAT_Class_Target'). Non-numeric or empty -> nodata_out.
    """
    out_path = Path(out_path)
    lookup_df = lookup_df.dropna(subset=[value_column]).copy()
    nodata_out = -9999 if nodata_out is None else int(nodata_out)

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        nodata_in = src.nodata
        # Build mapping: raster values (as stored) -> target code
        value_to_target = {}
        for _, row in lookup_df.iterrows():
            orig = row[value_column]
            try:
                orig_val = int(orig) if isinstance(orig, (int, float)) and float(orig) == int(float(orig)) else float(orig)
            except (ValueError, TypeError):
                continue
            new = row[target_column]
            if pd.isna(new) or str(new).strip() == "":
                continue
            try:
                new_val = int(float(new))
            except (ValueError, TypeError):
                continue
            value_to_target[orig_val] = new_val

        out_data = np.full(data.shape, nodata_out, dtype=np.int32)
        for orig_val, new_val in value_to_target.items():
            out_data[data == orig_val] = new_val
        if nodata_in is not None:
            out_data[data == nodata_in] = nodata_out
        out_meta = src.meta.copy()
        out_meta.update({"dtype": "int32", "nodata": nodata_out})
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_data, 1)
    return str(out_path)


def _pick_ssurgo_mukey_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("MUKEY", "mukey", "MapUnitKey", "mapunitkey"):
        if c in df.columns:
            return c
    return None


def _pick_usersoil_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("MUID", "ID", "SOIL_ID", "id"):
        if c in df.columns:
            return c
    return None


def _pick_usersoil_name_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("SNAM", "Name", "name", "SNAM_USDA"):
        if c in df.columns:
            return c
    return None


def _normalize_mukey(val: Any) -> Optional[int]:
    if pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def apply_field_specific_soil_overlay(
    soil_raster_path: str,
    fields_gdf: gpd.GeoDataFrame,
    ssurgo_df: pd.DataFrame,
    out_soil_path: Path,
    out_lookup_csv: Path,
    out_usersoil_csv: Path,
    soil_id_map: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """
    Rasterize crop field polygons to the soil raster grid, then assign new pixel IDs for each
    unique (Field_ID, MUKEY) inside fields. Outside fields, keep original soil (MUKEY) values.

    Writes:
      - Field-specific soil GeoTIFF (int32)
      - lookup_soil.csv: Raster_Value, Name
      - usersoil.csv: original SSURGO rows plus duplicated rows for new IDs
    """
    out_soil_path = Path(out_soil_path)
    out_lookup_csv = Path(out_lookup_csv)
    out_usersoil_csv = Path(out_usersoil_csv)

    mukey_col = _pick_ssurgo_mukey_col(ssurgo_df)
    id_col = _pick_usersoil_id_col(ssurgo_df)
    name_col = _pick_usersoil_name_col(ssurgo_df)

    mukey_to_name: Dict[int, str] = {}
    if name_col is not None:
        key_col = mukey_col if mukey_col is not None else id_col
        if key_col is not None:
            for _, row in ssurgo_df.iterrows():
                mk = _normalize_mukey(row[key_col])
                if mk is None:
                    continue
                nm = row[name_col]
                label = str(nm) if pd.notna(nm) else str(mk)
                if mk not in mukey_to_name:
                    mukey_to_name[mk] = label

    with rasterio.open(soil_raster_path) as src:
        soil_arr = src.read(1)
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        nodata_in = src.nodata
        meta = src.meta.copy()
        soil_bounds = src.bounds

    soil_int = np.rint(soil_arr).astype(np.int64, copy=False)
    if soil_id_map:
        mapped_soil = soil_int.copy()
        for raw_val, true_id in soil_id_map.items():
            mapped_soil[soil_int == int(raw_val)] = int(true_id)
        soil_int = mapped_soil

    fg = fields_gdf.copy().reset_index(drop=True)
    if fg.crs is None:
        raise ValueError("Crop fields GeoDataFrame must have a CRS set.")
    if crs is not None and fg.crs != crs:
        fg = fg.to_crs(crs)

    fg["Field_ID"] = np.arange(1, len(fg) + 1, dtype=np.int32)
    shapes = [
        (geom, int(fid))
        for geom, fid in zip(fg.geometry, fg["Field_ID"])
        if geom is not None and not geom.is_empty
    ]
    field_arr = rasterio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )

    valid_soil = np.ones(soil_arr.shape, dtype=bool)
    if nodata_in is not None and np.isfinite(nodata_in):
        valid_soil = soil_arr != nodata_in

    max_mukey = int(np.max(soil_int[valid_soil])) if valid_soil.any() else 0
    if not valid_soil.any():
        max_mukey = int(np.max(soil_int)) if soil_int.size else 0

    mask_field = field_arr > 0
    if nodata_in is not None and np.isfinite(nodata_in):
        mask_field &= soil_arr != nodata_in
    if not mask_field.any():
        field_pixels = int((field_arr > 0).sum())
        valid_soil_pixels = int(valid_soil.sum())
        overlap_pixels = int(mask_field.sum())
        field_bounds = None
        if len(fg) > 0:
            fb = fg.total_bounds
            field_bounds = {
                "left": float(fb[0]),
                "bottom": float(fb[1]),
                "right": float(fb[2]),
                "top": float(fb[3]),
            }
        res_x = float(abs(transform.a))
        res_y = float(abs(transform.e))
        debug_lines = [
            "No crop field polygons overlap the soil raster extent.",
            "",
            "Debug info:",
            f"- Soil raster path: {soil_raster_path}",
            f"- Soil CRS: {crs}",
            f"- Soil bounds: left={soil_bounds.left:.3f}, bottom={soil_bounds.bottom:.3f}, right={soil_bounds.right:.3f}, top={soil_bounds.top:.3f}",
            f"- Soil raster shape: height={height}, width={width}",
            f"- Soil raster resolution: x={res_x:.6g}, y={res_y:.6g}",
            f"- Soil nodata: {nodata_in}",
            f"- Valid soil pixels: {valid_soil_pixels}",
            f"- Crop fields CRS after reprojection: {fg.crs}",
            f"- Crop field feature count: {len(fg)}",
            f"- Rasterized field pixels: {field_pixels}",
            f"- Overlap pixels with valid soil: {overlap_pixels}",
        ]
        if field_bounds is not None:
            debug_lines.append(
                f"- Crop field bounds after reprojection: left={field_bounds['left']:.3f}, bottom={field_bounds['bottom']:.3f}, right={field_bounds['right']:.3f}, top={field_bounds['top']:.3f}"
            )
        raise ValueError("\n".join(debug_lines))

    f_flat = field_arr[mask_field]
    s_flat = soil_int[mask_field]
    pairs = np.column_stack([f_flat, s_flat])
    unique_pairs = np.unique(pairs, axis=0)

    combo_to_new: Dict[Tuple[int, int], int] = {}
    next_id = max_mukey + 1
    for row_pair in unique_pairs:
        fid, mkey = int(row_pair[0]), int(row_pair[1])
        combo_to_new[(fid, mkey)] = next_id
        next_id += 1

    out_arr = soil_int.astype(np.int32, copy=True)
    for (fid, mkey), new_val in combo_to_new.items():
        sel = (field_arr == fid) & (soil_int == mkey)
        out_arr[sel] = np.int32(new_val)

    meta.update({"dtype": "int32", "nodata": nodata_in})
    with rasterio.open(out_soil_path, "w", **meta) as dest:
        dest.write(out_arr, 1)

    newid_to_combo = {v: k for k, v in combo_to_new.items()}
    unique_vals = np.unique(out_arr)
    lookup_rows = []
    for v in unique_vals.tolist():
        if nodata_in is not None and np.isfinite(nodata_in) and int(v) == int(nodata_in):
            continue
        vid = int(v)
        if vid in newid_to_combo:
            fid, mkey = newid_to_combo[vid]
            label = f"Field_{fid}_Soil_{mkey}"
        else:
            label = mukey_to_name.get(vid, str(vid))
        lookup_rows.append({"Raster_Value": vid, "Name": label})

    lookup_df = pd.DataFrame(lookup_rows).sort_values("Raster_Value")
    lookup_df.to_csv(out_lookup_csv, index=False)

    usersoil_out = ssurgo_df.copy()
    if id_col is None or name_col is None:
        usersoil_out.to_csv(out_usersoil_csv, index=False)
        return {
            "soil_path": str(out_soil_path),
            "lookup_csv": str(out_lookup_csv),
            "usersoil_csv": str(out_usersoil_csv),
            "n_new_combos": len(combo_to_new),
            "usersoil_note": "ID/Name columns not found; SSURGO table copied without appended rows.",
        }

    new_rows: List[pd.Series] = []
    for (fid, mkey), new_id in combo_to_new.items():
        parent = pd.DataFrame()
        if mukey_col is not None:
            mask = ssurgo_df[mukey_col].apply(lambda x: _normalize_mukey(x) == mkey)
            parent = ssurgo_df.loc[mask]
        if parent.empty and id_col is not None:
            mask = ssurgo_df[id_col].apply(lambda x: _normalize_mukey(x) == mkey)
            parent = ssurgo_df.loc[mask]
        if parent.empty:
            continue
        row = parent.iloc[0].copy()
        try:
            row[id_col] = new_id
        except Exception:
            row[id_col] = str(new_id)
        row[name_col] = f"Field_{int(fid)}_Soil_{int(mkey)}"
        new_rows.append(row)

    if new_rows:
        usersoil_out = pd.concat([usersoil_out, pd.DataFrame(new_rows)], ignore_index=True)
    usersoil_out.to_csv(out_usersoil_csv, index=False)

    return {
        "soil_path": str(out_soil_path),
        "lookup_csv": str(out_lookup_csv),
        "usersoil_csv": str(out_usersoil_csv),
        "n_new_combos": len(combo_to_new),
    }
