from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import pandas as pd
import rasterio


def _format_bounds(bounds: Any) -> Dict[str, float]:
    """Return bounds as a serializable dictionary."""
    left, bottom, right, top = bounds
    return {
        "left": float(left),
        "bottom": float(bottom),
        "right": float(right),
        "top": float(top),
    }


def load_vector_from_zip(extracted_dir: Path) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Load a Shapefile from an extracted directory.

    Parameters
    ----------
    extracted_dir : Path
        Directory where the contents of the uploaded ZIP have been extracted.

    Returns
    -------
    (GeoDataFrame, dict)
        The vector dataset and basic metadata.
    """
    # Look for any .shp file in the extracted directory (non-recursive).
    shp_files = list(extracted_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No .shp file found in the uploaded ZIP.")

    # For now, take the first Shapefile found.
    shp_path = shp_files[0]
    gdf = gpd.read_file(shp_path)

    metadata: Dict[str, Any] = {
        "type": "vector",
        "driver": "ESRI Shapefile",
        "path": str(shp_path),
        "crs": str(gdf.crs) if gdf.crs else None,
        "feature_count": int(len(gdf)),
        "columns": list(gdf.columns),
        "geometry_type": gdf.geom_type.unique().tolist(),
        "bounds": _format_bounds(gdf.total_bounds),
    }

    return gdf, metadata


def get_numeric_columns(gdf: gpd.GeoDataFrame) -> list:
    """Return list of column names that are numeric (excluding geometry), for use as raster value source."""
    geom_name = gdf.geometry.name
    return [
        c for c in gdf.columns
        if c != geom_name and pd.api.types.is_numeric_dtype(gdf[c])
    ]


def load_vector_from_path(path: Path) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Load a vector layer from a path (directory with .shp or path to .shp)."""
    path = Path(path)
    if path.is_dir():
        shp_files = list(path.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"No .shp in {path}")
        shp_path = shp_files[0]
    else:
        shp_path = path
    gdf = gpd.read_file(shp_path)
    metadata: Dict[str, Any] = {
        "type": "vector",
        "driver": "ESRI Shapefile",
        "path": str(shp_path),
        "crs": str(gdf.crs) if gdf.crs else None,
        "feature_count": int(len(gdf)),
        "columns": list(gdf.columns),
        "geometry_type": gdf.geom_type.unique().tolist(),
        "bounds": _format_bounds(gdf.total_bounds),
    }
    return gdf, metadata


def load_raster(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Load a raster file and return its path and metadata.

    Rasterio's dataset objects are not easily serializable or sharable
    across Streamlit sessions, so this utility returns the file path
    plus a summary metadata dictionary. The map utilities can reopen
    the dataset from the path when needed.
    """
    path = Path(path)

    with rasterio.open(path) as src:
        res = src.res
        resolution_str = f"{float(res[0]):.4g} m" if res and len(res) >= 2 else None
        metadata: Dict[str, Any] = {
            "type": "raster",
            "driver": src.driver,
            "path": str(path),
            "crs": str(src.crs) if src.crs else None,
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "dtype": src.dtypes[0] if src.count > 0 else None,
            "bounds": _format_bounds(src.bounds),
            "transform": tuple(src.transform),
            "resolution": resolution_str,
        }

    # We return the path so downstream code can reopen the dataset.
    return str(path), metadata


def load_table(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a CSV file into a DataFrame and basic metadata."""
    path = Path(path)
    df = pd.read_csv(path)

    # For metadata we keep it light to avoid heavy introspection on large tables.
    metadata: Dict[str, Any] = {
        "type": "table",
        "path": str(path),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
    }

    return df, metadata

