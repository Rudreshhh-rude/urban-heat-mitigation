import os
import sys
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Polygon

# Ensure backend folder is in path for config
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CONFIG_DIR)

import config

def fetch_bengaluru_boundary():
    """
    Attempts to fetch the administrative boundary of Bengaluru from Nominatim.
    Falls back to a high-fidelity simplified polygon of the BBMP area if offline/failed.
    """
    print("[GIS] Querying Nominatim for Bruhat Bengaluru Mahanagara Palike (BBMP) boundary...")
    url = "https://nominatim.openstreetmap.org/search?q=Bengaluru&format=geojson&polygon_geojson=1"
    headers = {"User-Agent": "UrbanHeatMitigationAgent/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            
            # Look for polygons or multipolygons representing the admin boundary
            candidates = [f for f in features if f["geometry"]["type"] in ["Polygon", "MultiPolygon"]]
            if candidates:
                # Pick the most complex candidate polygon (highest number of coordinates)
                best_feature = None
                max_coords = 0
                for c in candidates:
                    geom = c["geometry"]
                    if geom["type"] == "Polygon":
                        num_coords = len(geom["coordinates"][0])
                    else:
                        num_coords = sum(len(poly[0]) for poly in geom["coordinates"])
                    
                    if num_coords > max_coords:
                        max_coords = num_coords
                        best_feature = c
                
                if best_feature:
                    print(f"[GIS] Success: Resolved OSM boundary: {best_feature['properties'].get('display_name')}")
                    return shape(best_feature["geometry"])
    except Exception as e:
        print(f"[GIS WARNING] API query failed, using offline fallback. Reason: {str(e)}")
    
    # Offline fallback municipal outline of Bengaluru (BBMP boundary vertices)
    print("[GIS] Constructing simplified operational zone envelope fallback...")
    bbmp_coords = [
        (77.46, 12.83), (77.48, 12.92), (77.45, 12.98), (77.48, 13.06),
        (77.52, 13.12), (77.59, 13.14), (77.65, 13.12), (77.72, 13.08),
        (77.76, 13.02), (77.74, 12.94), (77.71, 12.88), (77.65, 12.82),
        (77.58, 12.80), (77.52, 12.81), (77.46, 12.83)
    ]
    return Polygon(bbmp_coords)

def prune_grid():
    # 1. Ingest files
    geojson_path = config.GRID_GEOMETRY_PATH
    parquet_path = config.OUTPUT_PARQUET_PATH
    
    print(f"[GIS] Loading H3 Grid Geometries from: {geojson_path}")
    grid_gdf = gpd.read_file(geojson_path)
    initial_cell_count = len(grid_gdf)
    print(f"[GIS] Loaded {initial_cell_count} cell indices.")
    
    print(f"[GIS] Loading Parquet Feature matrix from: {parquet_path}")
    parquet_df = pd.read_parquet(parquet_path)
    
    # 2. Get city boundary polygon
    city_polygon = fetch_bengaluru_boundary()
    
    # 3. Intersect cells (WGS84 check)
    print("[GIS] Running spatial intersection check against city boundary...")
    # Keep cell if boundary intersects or centroid is inside municipal polygon
    keep_mask = grid_gdf.intersects(city_polygon) | grid_gdf.centroid.intersects(city_polygon)
    
    pruned_grid_gdf = grid_gdf[keep_mask].copy()
    pruned_cell_count = len(pruned_grid_gdf)
    pruned_cell_ids = pruned_grid_gdf["H3_Index_ID"].unique()
    
    print(f"[GIS] Spatial pruning complete. Reduced grid from {initial_cell_count} cells down to {pruned_cell_count} municipal cells.")
    
    # 4. Filter features dataset to match
    pruned_parquet_df = parquet_df[parquet_df["H3_Index_ID"].isin(pruned_cell_ids)].copy()
    print(f"[GIS] Feature matrix filtered: {len(parquet_df)} rows reduced to {len(pruned_parquet_df)} rows.")
    
    # 5. Save outputs
    print(f"[GIS] Saving pruned geometries back to: {geojson_path}")
    pruned_grid_gdf.to_file(geojson_path, driver="GeoJSON")
    
    print(f"[GIS] Saving pruned features back to: {parquet_path}")
    pruned_parquet_df.to_parquet(parquet_path, index=False)
    print("[GIS] Spatial pruning pipeline completed successfully!")

if __name__ == "__main__":
    prune_grid()
