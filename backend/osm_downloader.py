import requests
import geopandas as gpd
from shapely.geometry import Polygon
import h3
import config
from grid_generator import get_h3_api

def fetch_osm_buildings(bbox):
    """
    Fetches OSM building centroids using the Overpass API.
    Uses 'out center;' to keep response size lightweight and prevent gateway timeouts.
    """
    print("Fetching building data from OpenStreetMap Overpass API...")
    
    # Overpass Query (ways with building tags, returning center coordinates only)
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    overpass_query = f"""
    [out:json][timeout:180];
    (
      way["building"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
    );
    out center;
    """
    
    response = requests.post(overpass_url, data={'data': overpass_query}, headers={'User-Agent': 'UrbanHeatMitigationAgent/1.0'})
    
    if response.status_code != 200:
        raise RuntimeError(f"Overpass API failed with status code {response.status_code}: {response.text}")
        
    data = response.json()
    elements = data.get('elements', [])
    print(f"Retrieved {len(elements)} buildings from OSM.")
    return elements

def calculate_building_density(elements, h3_cells, resolution):
    """
    Calculates building density (footprint area / cell area) for each H3 cell.
    Points are mapped to cells using H3 index lookups.
    """
    funcs = get_h3_api()
    latlng_to_cell = funcs['latlng_to_cell']
    cell_to_boundary = funcs['cell_to_boundary']
    
    # Count buildings per H3 cell
    counts = {}
    for el in elements:
        center = el.get('center')
        if not center:
            continue
        lat, lon = center['lat'], center['lon']
        try:
            cell = latlng_to_cell(lat, lon, resolution)
            counts[cell] = counts.get(cell, 0) + 1
        except Exception:
            continue
            
    # Calculate UTM Area for each H3 cell
    print("Calculating cell metric areas in UTM Zone 43N...")
    density_map = {}
    
    # Estimate building density
    # Assume an average building footprint area of 120 square meters
    avg_building_footprint = 120.0 
    
    for h3_id in h3_cells:
        count = counts.get(h3_id, 0)
        if count == 0:
            density_map[h3_id] = 0.0
            continue
            
        # Get geometry and reproject to UTM CRS
        boundary = cell_to_boundary(h3_id)
        poly = Polygon([(pt[1], pt[0]) for pt in boundary])
        
        # Convert to GeoSeries to reproject and calculate area in sqm
        gs = gpd.GeoSeries([poly], crs=config.WGS84_CRS)
        gs_utm = gs.to_crs(config.UTM_CRS)
        cell_area_sqm = gs_utm.area.iloc[0]
        
        # Calculate density
        density = (count * avg_building_footprint) / cell_area_sqm
        density_map[h3_id] = min(1.0, float(density)) # Cap at 1.0 (100%)
        
    return density_map

if __name__ == "__main__":
    from grid_generator import generate_h3_grid
    
    # Test OSM building density calculator
    h3_cells = generate_h3_grid(config.BBOX, config.H3_RESOLUTION)
    try:
        buildings = fetch_osm_buildings(config.BBOX)
        density = calculate_building_density(buildings, h3_cells, config.H3_RESOLUTION)
        # Print a sample of non-zero densities
        non_zero = {k: v for k, v in density.items() if v > 0}
        print(f"Calculated density for {len(density)} cells. Non-zero cells: {len(non_zero)}.")
        if non_zero:
            sample_key = list(non_zero.keys())[0]
            print(f"Sample Cell ID: {sample_key}, Density: {non_zero[sample_key]:.4f}")
    except Exception as e:
        print(f"OSM download/calculation failed: {e}")
