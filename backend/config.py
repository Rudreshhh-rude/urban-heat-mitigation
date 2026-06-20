import os

# Configuration file for Bengaluru Urban Heat Mitigation

# Bounding box for study area (Bengaluru, India)
BBOX = {
    "min_lat": 12.85,
    "max_lat": 13.10,
    "min_lon": 77.50,
    "max_lon": 77.75
}

# Coordinate Reference Systems
WGS84_CRS = "EPSG:4326"
UTM_CRS = "EPSG:32643"  # UTM Zone 43N for Bengaluru region

# H3 Index settings
# Resolution 9: avg area ~0.1 km2, avg edge length ~174m
H3_RESOLUTION = 9

# Date range for analysis (April 2024 - peak pre-monsoon summer)
START_DATE = "2024-04-01"
END_DATE = "2024-04-30"

# Helper to load .env manually to avoid external dependencies
def load_dotenv(env_path):
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    os.environ[key] = val

# Load configuration relative to this file's location
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CONFIG_DIR, ".env"))

def get_resolved_path(env_key, default_filename):
    val = os.getenv(env_key, default_filename)
    if os.path.isabs(val):
        return val
    return os.path.abspath(os.path.join(CONFIG_DIR, val))

# Output paths dynamically loaded from env
GRID_GEOMETRY_PATH = get_resolved_path("GEOJSON_PATH", "bengaluru_h3_grid.geojson")
OUTPUT_PARQUET_PATH = get_resolved_path("PARQUET_PATH", "h3_city_features.parquet")
MODEL_PATH = get_resolved_path("MODEL_PATH", "piml_urban_model.pt")
SCALER_PATH = get_resolved_path("SCALER_PATH", "scaler.pkl")

