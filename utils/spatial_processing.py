"""
Spatial processing utilities for QSWAT+ preprocessing: HUC lookup,
mosaic, clip, reproject, and land use class extraction/reclassification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as rasterio_merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
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
