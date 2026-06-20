from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from contextlib import asynccontextmanager
import json
import pandas as pd
import geopandas as gpd
import os
import sys
import pickle
import torch
import numpy as np

# Ensure backend folder is in path for resolving local imports
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CONFIG_DIR)

import config
from train import UrbanThermalMLP
from optimizer import optimize_cell_intervention_generator

# Global states
model = None
scaler = None
geojson_payload = None
features_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, geojson_payload, features_df
    try:
        print("[INIT] Loading biophysical PIML model and scaler configurations...")
        # 1. Load model weights
        model = UrbanThermalMLP(input_dim=5)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()

        # 2. Load StandardScaler state
        with open(config.SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("[INIT] Model and scaling structures successfully cached.")

        # 3. Load GeoJSON Grid and Parquet features once on startup
        print("[INIT] Initializing GIS geometry cache and feature matrices...")
        grid_gdf = gpd.read_file(config.GRID_GEOMETRY_PATH)
        grid_lean = grid_gdf[["H3_Index_ID", "geometry"]]

        features_df = pd.read_parquet(config.OUTPUT_PARQUET_PATH)

        # 4. Clean and round attributes to minimize payload size
        df_clean = features_df[["H3_Index_ID", "LST", "NDVI", "Albedo", "Building_Density", "Air_Temp", "Humidity"]].copy()
        float_cols = ["LST", "NDVI", "Albedo", "Building_Density", "Air_Temp", "Humidity"]
        for col in float_cols:
            df_clean[col] = df_clean[col].round(4)

        # 5. Merge and pre-serialize to JSON string to guarantee instant GET responses
        merged_gdf = grid_lean.merge(df_clean, on="H3_Index_ID", how="inner")
        geojson_dict = json.loads(merged_gdf.to_json())
        geojson_payload = json.dumps(geojson_dict)
        print("[INIT] Unified spatial grid cached and serialized.")

    except Exception as e:
        print(f"[FATAL] Failed during startup asset loading: {str(e)}")
        raise e
        
    yield
    # Cleanup on shutdown if any
    print("[SHUTDOWN] Releasing cached variables...")
    model = None
    scaler = None
    geojson_payload = None
    features_df = None

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend workspace access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/grid")
async def get_grid():
    if geojson_payload is None:
        return Response(
            content=json.dumps({"error": "Service has not completed asset startup serialization"}),
            status_code=503,
            media_type="application/json"
        )
    # Return pre-serialized JSON payload directly
    return Response(content=geojson_payload, media_type="application/json")

@app.websocket("/api/optimize-live")
async def optimize_live(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connection established.")
    
    try:
        # Expecting H3 index payload from client
        data = await websocket.receive_text()
        try:
            payload = json.loads(data)
            h3_index = payload.get("h3_index")
        except json.JSONDecodeError:
            h3_index = data.strip()
            
        print(f"[WS] Initiating evolutionary cooling optimization request for cell: {h3_index}")
        
        if features_df is None or model is None or scaler is None:
            await websocket.send_json({"error": "Optimization backend components not initialized"})
            await websocket.close()
            return
            
        # Look up cell row in cached dataset
        cell_rows = features_df[features_df["H3_Index_ID"] == h3_index]
        if cell_rows.empty:
            await websocket.send_json({"error": f"H3 Index {h3_index} not found in features dataset"})
            await websocket.close()
            return
            
        cell_data = cell_rows.iloc[0]
        
        # Run NSGA-II GA generator loop (50 generations, 100 population size)
        generator = optimize_cell_intervention_generator(
            cell_data=cell_data,
            model=model,
            scaler=scaler,
            generations=50,
            pop_size=100
        )
        
        for item in generator:
            if isinstance(item, tuple):
                # Intermediate generation telemetry frame: [generation_id, best_cooling_delta, current_pareto_count]
                gen_id, best_cooling, pareto_count = item
                await websocket.send_json([gen_id, round(best_cooling, 4), pareto_count])
            else:
                # Final Pareto front list of dict solutions
                # Ensure values are rounded for transmission efficiency
                clean_pareto = []
                for sol in item:
                    clean_pareto.append({
                        "delta_ndvi": round(sol["delta_ndvi"], 4),
                        "delta_albedo": round(sol["delta_albedo"], 4),
                        "lst_drop": round(sol["lst_drop"], 4),
                        "cost": round(sol["cost"], 4)
                    })
                await websocket.send_json({
                    "status": "complete",
                    "pareto_front": clean_pareto
                })
                
        print(f"[WS] Evolutionary optimization completed successfully for cell: {h3_index}")

    except WebSocketDisconnect:
        print("[WS] Client disconnected cleanly.")
    except Exception as e:
        print(f"[WS ERROR] Exception in WebSocket connection: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
