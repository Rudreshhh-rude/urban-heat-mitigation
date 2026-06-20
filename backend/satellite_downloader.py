import time
import numpy as np
import pystac_client
import planetary_computer
import rasterio
from rasterio.warp import transform_bounds, transform
from rasterio.windows import from_bounds
import config

def get_stac_catalog():
    """Returns a signed STAC catalog instance from Microsoft Planetary Computer."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )
    return catalog

def find_cleanest_item(catalog, collection, bbox, start_date, end_date, max_cloud=20):
    """
    Searches the catalog for the least cloudy scene within the bbox and date range.
    """
    print(f"Searching for cleanest scene in STAC collection: {collection}...")
    search = catalog.search(
        collections=[collection],
        bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    )
    items = list(search.item_collection())
    
    if not items:
        # Relax cloud cover constraint if nothing is found
        print(f"No scenes found under {max_cloud}% cloud cover. Relaxing constraint to 50%...")
        search = catalog.search(
            collections=[collection],
            bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": 50}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]
        )
        items = list(search.item_collection())
        
    if not items:
        # Relax completely
        print("No scenes found under 50% cloud cover. Searching all items...")
        search = catalog.search(
            collections=[collection],
            bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
            datetime=f"{start_date}/{end_date}",
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]
        )
        items = list(search.item_collection())
        
    if not items:
        raise ValueError(f"No STAC items found for collection {collection} in the configured timeframe.")
        
    cleanest_item = items[0]
    print(f"Found cleanest item {cleanest_item.id} with cloud cover: {cleanest_item.properties.get('eo:cloud_cover', 'unknown')}%")
    return cleanest_item

def read_raster_window(asset_url, bbox, retries=3):
    """
    Reads only the bounding box window from a remote cloud-optimized GeoTIFF.
    Uses rasterio's virtual file system /vsicurl/ to preserve memory and disk space.
    """
    for attempt in range(retries):
        try:
            with rasterio.open(asset_url) as src:
                # Reproject WGS84 bbox to raster's local coordinate system (CRS)
                left, bottom, right, top = transform_bounds(
                    config.WGS84_CRS,
                    src.crs,
                    bbox["min_lon"],
                    bbox["min_lat"],
                    bbox["max_lon"],
                    bbox["max_lat"]
                )
                
                # Get window and clamp to raster boundaries
                window = from_bounds(left, bottom, right, top, src.transform).round()
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                # Read the array data
                data = src.read(1, window=window).astype(np.float32)
                return src.transform, src.crs, window, data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed reading remote raster: {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e

def extract_values_at_centroids(src_transform, src_crs, window, data, lats, lons):
    """
    Extracts the values of the raster at the specified lat/lon coordinates.
    """
    # Transform lat/lons to raster CRS
    xs, ys = transform(config.WGS84_CRS, src_crs.to_string(), lons, lats)
    
    # Get pixel indices relative to full raster
    rows, cols = rasterio.transform.rowcol(src_transform, xs, ys)
    
    # Convert to relative pixel indices inside the window
    relative_rows = np.array(rows) - int(window.row_off)
    relative_cols = np.array(cols) - int(window.col_off)
    
    extracted_vals = []
    for r, c in zip(relative_rows, relative_cols):
        if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
            extracted_vals.append(data[r, c])
        else:
            extracted_vals.append(np.nan)
            
    return np.array(extracted_vals)

def process_sentinel_data(catalog, bbox, start_date, end_date, h3_gdf):
    """
    Retrieves Sentinel-2 data and computes cloud-masked NDVI and Albedo.
    """
    try:
        item = find_cleanest_item(catalog, "sentinel-2-l2a", bbox, start_date, end_date)
    except Exception as e:
        print(f"Warning: Sentinel-2 search failed: {e}. Falling back to default values.")
        h3_gdf["NDVI"] = 0.3
        h3_gdf["Albedo"] = 0.15
        return h3_gdf
        
    print("Extracting Sentinel-2 bands for NDVI and Albedo...")
    lats = h3_gdf["centroid_lat"].tolist()
    lons = h3_gdf["centroid_lon"].tolist()
    
    # Bands needed: B02, B04, B08, B11, B12, and SCL (scene classification)
    bands_needed = ["B02", "B04", "B08", "B11", "B12", "SCL"]
    band_data = {}
    
    for band in bands_needed:
        asset = item.assets.get(band)
        if not asset:
            # Fallback if B08 is named differently (e.g. B8A)
            if band == "B08":
                asset = item.assets.get("B8A")
            if not asset:
                raise ValueError(f"Could not find asset {band} in Sentinel-2 item.")
                
        print(f"Reading band {band} from COG...")
        transform_obj, crs_obj, window_obj, data_arr = read_raster_window(asset.href, bbox)
        
        # Extract values for H3 centroids
        vals = extract_values_at_centroids(transform_obj, crs_obj, window_obj, data_arr, lats, lons)
        band_data[band] = vals

    # Cloud masking using Scene Classification Layer (SCL)
    # SCL codes: 3 = shadow, 7 = low prob cloud, 8 = med prob cloud, 9 = high prob cloud, 10 = cirrus
    cloudy_mask = np.isin(band_data["SCL"], [3, 7, 8, 9, 10])
    print(f"SCL masked out {np.sum(cloudy_mask)} cloudy/shadowed centroids out of {len(h3_gdf)} cells.")

    # Calculate NDVI: (B08 - B04) / (B08 + B04)
    b04 = band_data["B04"]
    b08 = band_data["B08"]
    
    # Division protection
    denom = b08 + b04
    denom[denom == 0] = np.nan
    ndvi = (b08 - b04) / denom
    
    # Apply cloud mask
    ndvi[cloudy_mask] = np.nan
    # Clip NDVI to valid range [-1.0, 1.0]
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    # Calculate Broadband Albedo (Sentinel-2 values are stored as integers, scale by 0.0001)
    # Formula: Albedo = 0.356*B2 + 0.130*B4 + 0.373*B8 + 0.085*B11 + 0.072*B12
    scale = 0.0001
    b02_s = band_data["B02"] * scale
    b04_s = band_data["B04"] * scale
    b08_s = band_data["B08"] * scale
    b11_s = band_data["B11"] * scale
    b12_s = band_data["B12"] * scale
    
    albedo = 0.356*b02_s + 0.130*b04_s + 0.373*b08_s + 0.085*b11_s + 0.072*b12_s
    albedo[cloudy_mask] = np.nan
    albedo = np.clip(albedo, 0.0, 1.0) # Albedo range [0, 1]
    
    h3_gdf["NDVI"] = ndvi
    h3_gdf["Albedo"] = albedo
    
    return h3_gdf

def process_landsat_data(catalog, bbox, start_date, end_date, h3_gdf):
    """
    Retrieves Landsat 8 thermal band and computes Land Surface Temperature (LST).
    """
    try:
        item = find_cleanest_item(catalog, "landsat-c2-l2", bbox, start_date, end_date)
    except Exception as e:
        print(f"Warning: Landsat-8 search failed: {e}. Falling back to default LST values.")
        h3_gdf["LST"] = 35.0
        return h3_gdf
        
    print("Extracting Landsat-8 bands for LST...")
    lats = h3_gdf["centroid_lat"].tolist()
    lons = h3_gdf["centroid_lon"].tolist()
    
    # Thermal band (Band 10) is named lwir11 or similar in Landsat 8 STAC
    thermal_key = [k for k in item.assets.keys() if "lwir11" in k or "thermal" in k or "st_b10" in k]
    if not thermal_key:
        raise ValueError("Could not locate Landsat 8 thermal band asset.")
    thermal_key = thermal_key[0]
    
    qa_key = [k for k in item.assets.keys() if "qa_pixel" in k]
    if not qa_key:
        raise ValueError("Could not locate Landsat 8 QA Pixel asset.")
    qa_key = qa_key[0]
    
    print(f"Reading thermal band {thermal_key} from COG...")
    t_trans, t_crs, t_win, t_data = read_raster_window(item.assets[thermal_key].href, bbox)
    thermal_dn = extract_values_at_centroids(t_trans, t_crs, t_win, t_data, lats, lons)
    
    print(f"Reading QA pixel band {qa_key} from COG...")
    q_trans, q_crs, q_win, q_data = read_raster_window(item.assets[qa_key].href, bbox)
    qa_pixel = extract_values_at_centroids(q_trans, q_crs, q_win, q_data, lats, lons)
    
    # Cloud masking from qa_pixel:
    # Bit 3 = Dilated Cloud, Bit 4 = Cloud Shadow, Bit 5 = Cloud
    qa_int = qa_pixel.astype(np.uint16)
    cloudy_mask = ((qa_int & (1 << 3)) != 0) | ((qa_int & (1 << 4)) != 0) | ((qa_int & (1 << 5)) != 0)
    print(f"Landsat QA Pixel masked out {np.sum(cloudy_mask)} cloudy centroids out of {len(h3_gdf)} cells.")
    
    # Calculate Land Surface Temperature in Celsius
    # Landsat 8 Level-2 Surface Temperature bands store scaled Kelvin temperatures:
    # Kelvin = (DN * 0.00341802) + 149.0
    # Celsius = Kelvin - 273.15
    kelvin = (thermal_dn * 0.00341802) + 149.0
    lst_c = kelvin - 273.15
    
    # Mask fill values (e.g. 0 DN which maps to -124C or invalid numbers)
    invalid_mask = (thermal_dn == 0) | (lst_c < -10.0) | (lst_c > 65.0) | cloudy_mask
    lst_c[invalid_mask] = np.nan
    
    h3_gdf["LST"] = lst_c
    return h3_gdf

if __name__ == "__main__":
    from grid_generator import generate_h3_grid, create_geodataframe
    h3_cells = generate_h3_grid(config.BBOX, config.H3_RESOLUTION)
    gdf = create_geodataframe(h3_cells)
    
    catalog = get_stac_catalog()
    try:
        gdf = process_sentinel_data(catalog, config.BBOX, config.START_DATE, config.END_DATE, gdf)
        gdf = process_landsat_data(catalog, config.BBOX, config.START_DATE, config.END_DATE, gdf)
        print("Sample extracted satellite columns:")
        print(gdf[["H3_Index_ID", "NDVI", "Albedo", "LST"]].head(10))
    except Exception as e:
        print(f"Satellite extraction failed: {e}")
