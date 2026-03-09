from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import leafmap.foliumap as leafmap
import pandas as pd
import rasterio


def _center_from_bounds(bounds: Dict[str, float]) -> Tuple[float, float]:
    """Compute a reasonable map center from bounds."""
    return (
        (bounds["bottom"] + bounds["top"]) / 2.0,
        (bounds["left"] + bounds["right"]) / 2.0,
    )


def create_map(
    data: Any,
    metadata: Dict[str, Any],
    default_zoom: int = 8,
) -> leafmap.Map:
    """Create a Leafmap object visualizing the uploaded dataset.

    Parameters
    ----------
    data : Any
        - GeoDataFrame for vector data.
        - File path (str) for raster data.
        - DataFrame for tables (no spatial visualization yet).
    metadata : dict
        Metadata dictionary created by the file handlers.
    default_zoom : int, optional
        Default zoom level when computing initial view, by default 8.

    Returns
    -------
    leafmap.Map
        Configured interactive map containing the dataset, if applicable.
    """
    m = leafmap.Map(
        basemap="OpenStreetMap",
        draw_control=False,
        measure_control=False,
        fullscreen_control=True,
    )

    data_type = metadata.get("type")

    if data_type == "vector" and isinstance(data, gpd.GeoDataFrame):
        bounds = metadata.get("bounds")
        if bounds:
            center = _center_from_bounds(bounds)
            m.set_center(lon=center[1], lat=center[0], zoom=default_zoom)

        m.add_gdf(
            data,
            layer_name="Vector layer",
            zoom_to_layer=True,
        )

    elif data_type == "raster" and isinstance(data, str):
        bounds = metadata.get("bounds")
        if bounds:
            center = _center_from_bounds(bounds)
            m.set_center(lon=center[1], lat=center[0], zoom=default_zoom)

        # Leafmap can display local rasters directly using rasterio.
        raster_path: str = data
        with rasterio.open(raster_path) as src:
            m.add_raster(
                raster_path,
                layer_name="Raster layer",
                colormap="viridis",
            )

    else:
        # For pure tables (no geometry) or unknown types, we just
        # display a basemap centered on a default location.
        m.set_center(lon=0.0, lat=0.0, zoom=2)

    return m


def _sanitize_gdf_for_folium(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert non-JSON-serializable columns (e.g. Timestamps) to strings."""
    gdf = gdf.copy()
    for col in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(str)
    return gdf


def create_multi_layer_map(
    layers: List[Tuple[Any, Dict[str, Any], str]],
    huc_gdf: Optional[gpd.GeoDataFrame] = None,
    default_zoom: int = 8,
) -> leafmap.Map:
    """
    Build a map with multiple overlay layers and an optional HUC boundary.

    Parameters
    ----------
    layers : list of (data, metadata, layer_name)
        Each tuple: data (GeoDataFrame or raster path str), metadata dict, display name.
    huc_gdf : GeoDataFrame, optional
        Optional HUC boundary to draw first (under other layers).
    default_zoom : int
        Zoom level when centering from bounds.

    Returns
    -------
    leafmap.Map
    """
    m = leafmap.Map(
        basemap="OpenStreetMap",
        draw_control=False,
        measure_control=False,
        fullscreen_control=True,
    )

    first_bounds = None

    # Add HUC boundary first (reproject to WGS84 for Leaflet if needed)
    if huc_gdf is not None and len(huc_gdf) > 0:
        huc_to_draw = _sanitize_gdf_for_folium(huc_gdf)
        try:
            if huc_to_draw.crs is not None and huc_to_draw.crs.to_epsg() != 4326:
                huc_to_draw = huc_to_draw.to_crs(4326)
        except Exception:
            # If CRS conversion fails, fall back to original geometry
            huc_to_draw = _sanitize_gdf_for_folium(huc_gdf)

        # Draw HUC boundary in red so it stands out over rasters.
        m.add_gdf(
            huc_to_draw,
            layer_name="HUC watershed",
            zoom_to_layer=False,
            style={"color": "red", "weight": 3, "fillOpacity": 0},
        )
        first_bounds = _bounds_from_gdf(huc_to_draw)

    for data, metadata, layer_name in layers:
        dtype = metadata.get("type")
        bounds = metadata.get("bounds")

        if dtype == "vector" and isinstance(data, gpd.GeoDataFrame):
            gdf_to_draw = _sanitize_gdf_for_folium(data)
            try:
                if gdf_to_draw.crs is not None and gdf_to_draw.crs.to_epsg() != 4326:
                    gdf_to_draw = gdf_to_draw.to_crs(4326)
            except Exception:
                gdf_to_draw = _sanitize_gdf_for_folium(data)

            if first_bounds is None:
                first_bounds = _bounds_from_gdf(gdf_to_draw)
            m.add_gdf(gdf_to_draw, layer_name=layer_name, zoom_to_layer=False)

        elif dtype == "raster" and isinstance(data, str):
            if bounds and first_bounds is None:
                first_bounds = bounds
            m.add_raster(data, layer_name=layer_name, colormap="viridis")

    if first_bounds:
        center = _center_from_bounds(first_bounds)
        m.set_center(lon=center[1], lat=center[0], zoom=default_zoom)
    else:
        m.set_center(lon=0.0, lat=0.0, zoom=2)

    # Ensure users can toggle layers on and off.
    m.add_layer_control()

    return m


def _bounds_from_gdf(gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    """Return bounds dict from a GeoDataFrame's total_bounds."""
    b = gdf.total_bounds
    return {"left": float(b[0]), "bottom": float(b[1]), "right": float(b[2]), "top": float(b[3])}

