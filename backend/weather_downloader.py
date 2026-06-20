import requests
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import config

def fetch_era5_weather_grid(bbox, start_date, end_date):
    """
    Queries Open-Meteo's historical ERA5 API for a 5x5 grid of points
    covering the study area, returning temperature, relative humidity, and wind speed.
    """
    print("Fetching weather data from Open-Meteo ERA5 API...")
    
    # Generate 5x5 coordinate grid
    lat_grid = np.linspace(bbox["min_lat"], bbox["max_lat"], 5)
    lon_grid = np.linspace(bbox["min_lon"], bbox["max_lon"], 5)
    
    lats = []
    lons = []
    for lat in lat_grid:
        for lon in lon_grid:
            lats.append(lat)
            lons.append(lon)
            
    lat_str = ",".join([f"{l:.4f}" for l in lats])
    lon_str = ",".join([f"{l:.4f}" for l in lons])
    
    url = (
        f"https://archive-api.open-meteo.com/v1/era5"
        f"?latitude={lat_str}&longitude={lon_str}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    
    response = requests.get(url, headers={'User-Agent': 'UrbanHeatMitigationAgent/1.0'})
    if response.status_code != 200:
        raise RuntimeError(f"Open-Meteo ERA5 API failed with status: {response.status_code}")
        
    data = response.json()
    
    # Open-Meteo returns a list of dicts when querying multiple coordinates,
    # or a single dict if only 1 location is queried.
    results = data if isinstance(data, list) else [data]
    
    weather_data = []
    for idx, res in enumerate(results):
        lat = lats[idx]
        lon = lons[idx]
        hourly = res.get("hourly", {})
        
        # Calculate monthly average from hourly lists
        temps = hourly.get("temperature_2m", [])
        humids = hourly.get("relative_humidity_2m", [])
        winds = hourly.get("wind_speed_10m", [])
        
        if not temps or not humids:
            continue
            
        mean_temp = np.mean(temps)
        mean_humid = np.mean(humids)
        mean_wind = np.mean(winds)
        
        weather_data.append({
            "lat": lat,
            "lon": lon,
            "air_temp": mean_temp,
            "humidity": mean_humid,
            "wind_speed": mean_wind
        })
        
    print(f"Retrieved ERA5 parameters for {len(weather_data)} grid locations.")
    return pd.DataFrame(weather_data)

def interpolate_weather_to_h3(weather_df, h3_gdf):
    """
    Interpolates weather grid points to all H3 centroids in the GeoDataFrame.
    Uses SciPy's griddata interpolation.
    """
    print("Interpolating weather features to H3 cell centroids...")
    grid_coords = weather_df[["lon", "lat"]].values
    
    target_coords = h3_gdf[["centroid_lon", "centroid_lat"]].values
    
    # Interpolate each variable
    air_temp_interp = griddata(grid_coords, weather_df["air_temp"].values, target_coords, method='linear')
    humidity_interp = griddata(grid_coords, weather_df["humidity"].values, target_coords, method='linear')
    wind_speed_interp = griddata(grid_coords, weather_df["wind_speed"].values, target_coords, method='linear')
    
    # Handle any edge/NaN values using nearest neighbor fallback
    nan_mask = np.isnan(air_temp_interp)
    if np.any(nan_mask):
        print("Using nearest-neighbor fallback for boundary H3 cells...")
        fallback_temp = griddata(grid_coords, weather_df["air_temp"].values, target_coords[nan_mask], method='nearest')
        fallback_humid = griddata(grid_coords, weather_df["humidity"].values, target_coords[nan_mask], method='nearest')
        fallback_wind = griddata(grid_coords, weather_df["wind_speed"].values, target_coords[nan_mask], method='nearest')
        
        air_temp_interp[nan_mask] = fallback_temp
        humidity_interp[nan_mask] = fallback_humid
        wind_speed_interp[nan_mask] = fallback_wind
        
    h3_gdf["Air_Temp"] = air_temp_interp
    h3_gdf["Humidity"] = humidity_interp
    h3_gdf["Wind_Speed"] = wind_speed_interp
    
    return h3_gdf

if __name__ == "__main__":
    from grid_generator import generate_h3_grid, create_geodataframe
    h3_cells = generate_h3_grid(config.BBOX, config.H3_RESOLUTION)
    gdf = create_geodataframe(h3_cells)
    
    try:
        weather_df = fetch_era5_weather_grid(config.BBOX, config.START_DATE, config.END_DATE)
        gdf = interpolate_weather_to_h3(weather_df, gdf)
        print("Sample interpolated weather columns:")
        print(gdf[["H3_Index_ID", "Air_Temp", "Humidity", "Wind_Speed"]].head())
    except Exception as e:
        print(f"Weather downloading/interpolation failed: {e}")
