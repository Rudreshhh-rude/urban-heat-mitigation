import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import h3
import config

def get_h3_api():
    """Returns functions compatible with installed H3 version (v3 or v4)."""
    h3_funcs = {}
    # latlng to cell
    if hasattr(h3, 'latlng_to_cell'):
        h3_funcs['latlng_to_cell'] = h3.latlng_to_cell
    elif hasattr(h3, 'geo_to_h3'):
        h3_funcs['latlng_to_cell'] = h3.geo_to_h3
    else:
        raise ImportError("Could not find H3 conversion function.")
    
    # cell to latlng
    if hasattr(h3, 'cell_to_latlng'):
        h3_funcs['cell_to_latlng'] = h3.cell_to_latlng
    elif hasattr(h3, 'h3_to_geo'):
        h3_funcs['cell_to_latlng'] = h3.h3_to_geo
    else:
        raise ImportError("Could not find H3 cell center function.")

    # cell to boundary
    if hasattr(h3, 'cell_to_boundary'):
        h3_funcs['cell_to_boundary'] = h3.cell_to_boundary
    elif hasattr(h3, 'h3_to_geo_boundary'):
        h3_funcs['cell_to_boundary'] = h3.h3_to_geo_boundary
    else:
        raise ImportError("Could not find H3 boundary function.")
        
    return h3_funcs

def generate_h3_grid(bbox, resolution):
    """
    Generates all H3 cells covering the bounding box.
    Uses point sampling to cover the box, which works robustly across versions.
    """
    funcs = get_h3_api()
    latlng_to_cell = funcs['latlng_to_cell']
    
    min_lat = bbox["min_lat"]
    max_lat = bbox["max_lat"]
    min_lon = bbox["min_lon"]
    max_lon = bbox["max_lon"]
    
    # Step size of approx 100 meters (0.001 degrees)
    lat_steps = np.arange(min_lat, max_lat + 0.001, 0.001)
    lon_steps = np.arange(min_lon, max_lon + 0.001, 0.001)
    
    h3_cells = set()
    for lat in lat_steps:
        for lon in lon_steps:
            h3_id = latlng_to_cell(lat, lon, resolution)
            h3_cells.add(h3_id)
            
    print(f"Generated {len(h3_cells)} unique H3 cells at resolution {resolution} for bounding box.")
    return list(h3_cells)

def create_geodataframe(h3_ids):
    """
    Creates a GeoDataFrame containing H3 polygons and centroids.
    """
    funcs = get_h3_api()
    cell_to_boundary = funcs['cell_to_boundary']
    cell_to_latlng = funcs['cell_to_latlng']
    
    geometries = []
    centroids_lat = []
    centroids_lon = []
    
    for h3_id in h3_ids:
        # Get boundary (lng, lat coordinates)
        # H3 returns (lat, lng) tuples for boundary. We swap to (lng, lat) for Shapely.
        boundary = cell_to_boundary(h3_id)
        poly = Polygon([(pt[1], pt[0]) for pt in boundary])
        geometries.append(poly)
        
        # Get centroid
        lat, lon = cell_to_latlng(h3_id)
        centroids_lat.append(lat)
        centroids_lon.append(lon)
        
    gdf = gpd.GeoDataFrame({
        'H3_Index_ID': h3_ids,
        'centroid_lat': centroids_lat,
        'centroid_lon': centroids_lon
    }, geometry=geometries, crs=config.WGS84_CRS)
    
    return gdf

if __name__ == "__main__":
    print("Initializing Unified H3 Grid...")
    h3_cells = generate_h3_grid(config.BBOX, config.H3_RESOLUTION)
    gdf = create_geodataframe(h3_cells)
    
    # Save to file
    gdf.to_file(config.GRID_GEOMETRY_PATH, driver="GeoJSON")
    print(f"Saved H3 Grid GeoJSON to: {config.GRID_GEOMETRY_PATH}")
