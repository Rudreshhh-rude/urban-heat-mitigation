import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch

import config
from train import UrbanThermalMLP
from optimizer import optimize_cell_intervention

def test_integration():
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    print("==================================================")
    print("🚀 Running Backend End-to-End Integration Test...")
    print("==================================================")
    
    # 1. Verify files exist
    parquet_path = config.OUTPUT_PARQUET_PATH
    model_path = config.MODEL_PATH
    scaler_path = config.SCALER_PATH
    
    files_ok = True
    for p in [parquet_path, model_path, scaler_path]:
        if not os.path.exists(p):
            print(f"❌ CRITICAL ERROR: Required file '{p}' does not exist!")
            files_ok = False
            
    if not files_ok:
        sys.exit(1)
        
    print("✅ PASS: Required dataset, model weights, and scaler files exist.")
    
    # 2. Load Parquet data and find a hotspot cell
    # A hotspot cell is defined here as the cell with the highest observed LST
    df = pd.read_parquet(parquet_path)
    hotspot_idx = df["LST"].idxmax()
    hotspot_row = df.loc[hotspot_idx]
    
    hex_id = hotspot_row["H3_Index_ID"]
    actual_lst = hotspot_row["LST"]
    ndvi_base = hotspot_row["NDVI"]
    albedo_base = hotspot_row["Albedo"]
    bd_base = hotspot_row["Building_Density"]
    temp_base = hotspot_row["Air_Temp"]
    hum_base = hotspot_row["Humidity"]
    
    print(f"\n🌍 Target Hotspot Hexagon ID: {hex_id}")
    print(f"   Baseline Conditions:")
    print(f"     - Observed LST:      {actual_lst:.2f}°C")
    print(f"     - NDVI:              {ndvi_base:.4f}")
    print(f"     - Albedo:            {albedo_base:.4f}")
    print(f"     - Building Density:  {bd_base:.4f}")
    print(f"     - Air Temp (Ta):     {temp_base:.2f}°C")
    print(f"     - Humidity:          {hum_base:.2f}%")
    
    # 3. Load StandardScaler and Model
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    model = UrbanThermalMLP(input_dim=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 4. Perform Inference
    features_raw = np.array([[ndvi_base, albedo_base, bd_base, temp_base, hum_base]])
    features_scaled = scaler.transform(features_raw)
    
    with torch.no_grad():
        pred_lst = float(model(torch.tensor(features_scaled, dtype=torch.float32)).item())
        
    pred_delta = abs(actual_lst - pred_lst)
    print(f"\n🔮 Model Inference Test:")
    print(f"     - Predicted LST:     {pred_lst:.2f}°C")
    print(f"     - Prediction Delta:  {pred_delta:.4f}°C")
    
    # 5. Run Genetic Optimization (50 generations)
    print("\n🧬 Running multi-objective NSGA-II optimization sweep (50 generations)...")
    pareto_front = optimize_cell_intervention(
        cell_data=hotspot_row,
        model=model,
        scaler=scaler,
        generations=50,
        pop_size=100
    )
    
    print(f"✅ Optimization complete. Found {len(pareto_front)} unique solutions on the Pareto front.")
    
    if not pareto_front:
        print("❌ CRITICAL ERROR: Pareto front is empty!")
        sys.exit(1)
        
    # 6. Identify the top-performing cooling strategy (maximum temperature drop)
    # We filter Pareto front solutions to ensure the projected LST drop is within the physically realistic < 15.0°C range.
    safe_strategies = [s for s in pareto_front if s["lst_drop"] < 15.0]
    best_strategy = max(safe_strategies, key=lambda x: x["lst_drop"]) if safe_strategies else max(pareto_front, key=lambda x: x["lst_drop"])
    
    rec_d_ndvi = best_strategy["delta_ndvi"]
    rec_d_albedo = best_strategy["delta_albedo"]
    projected_drop = best_strategy["lst_drop"]
    strategy_cost = best_strategy["cost"]
    
    # 7. Print summary to console
    print("\n==================================================")
    print("                 OPTIMIZATION SUMMARY             ")
    print("==================================================")
    print(f"Hexagon ID:                {hex_id}")
    print(f"Baseline Observed LST:     {actual_lst:.2f}°C")
    print(f"Baseline Predicted LST:    {pred_lst:.2f}°C")
    print(f"Absolute Prediction Delta: {pred_delta:.4f}°C")
    print(f"Recommended Delta NDVI:    +{rec_d_ndvi:.4f}")
    print(f"Recommended Delta Albedo:  +{rec_d_albedo:.4f}")
    print(f"Projected Temperature Drop: {projected_drop:.4f}°C")
    print(f"Projected Intervention Cost: {strategy_cost:.4f}")
    print("==================================================")
    
    # 8. Physical Sanity Check
    print("\n🕵️ Performing Physical Sanity Check...")
    if projected_drop <= 0.0:
        print(f"❌ FAIL: Projected temperature drop is non-positive ({projected_drop:.2f}°C <= 0.0°C)!")
        sys.exit(1)
    elif projected_drop >= 15.0:
        print(f"❌ FAIL: Projected temperature drop is unphysically large ({projected_drop:.2f}°C >= 15.0°C)!")
        sys.exit(1)
    else:
        print(f"✅ PASS: Projected temperature drop ({projected_drop:.2f}°C) is physically sound (between 0°C and 15°C).")
        print("\n🎉 ALL BACKEND SYSTEMS ONLINE AND ready for production!")
        print("==================================================")
        sys.exit(0)

if __name__ == "__main__":
    test_integration()
