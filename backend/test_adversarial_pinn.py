import os
import sys
import pickle
import numpy as np
import torch

# Ensure backend folder is in path for imports
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CONFIG_DIR)

import config
from train import UrbanThermalMLP

def run_adversarial_audit():
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    print("==================================================")
    print("🕵️ Initiating Biophysical PINN Boundary Audit...")
    print("==================================================")

    # 1. Verify model artifacts exist
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        print("ERROR: Model weights or scaler parameters missing. Run train.py first.")
        sys.exit(1)

    # 2. Load model and scaler
    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = UrbanThermalMLP(input_dim=5)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print(" Model and Scaler structures successfully loaded into memory.")

    # 3. Define extreme boundary conditions
    # Features: [NDVI, Albedo, Building_Density, Air_Temp, Humidity]
    scenarios = {
        "Extreme Hot Concrete Void": {
            "NDVI": 0.0,
            "Albedo": 0.95,
            "Building_Density": 1.0,
            "Air_Temp": 50.0,
            "Humidity": 5.0,
            "expectation": "LST should be very high, but must remain physically bounded (> 35°C)."
        },
        "Supercooled Vegetated Forest": {
            "NDVI": 0.95,
            "Albedo": 0.05,
            "Building_Density": 0.0,
            "Air_Temp": 5.0,
            "Humidity": 95.0,
            "expectation": "LST should be low, but positive (> 0°C under moderate solar radiation)."
        },
        "Frozen Reflective Tundra": {
            "NDVI": 0.0,
            "Albedo": 0.98,
            "Building_Density": 0.0,
            "Air_Temp": -15.0,
            "Humidity": 2.0,
            "expectation": "Sub-zero conditions. Check if model outputs mathematically impossible values."
        },
        "Extreme Concrete Greenhouse": {
            "NDVI": 0.0,
            "Albedo": 0.02, # black asphalt roofs
            "Building_Density": 1.0,
            "Air_Temp": 45.0,
            "Humidity": 90.0,
            "expectation": "Extreme hot thermal greenhouse anomaly."
        }
    }

    print("\nExecuting forward inferences on extreme input tensors...")
    print("--------------------------------------------------")

    for name, params in scenarios.items():
        raw_features = np.array([[
            params["NDVI"],
            params["Albedo"],
            params["Building_Density"],
            params["Air_Temp"],
            params["Humidity"]
        ]])
        
        # Transform features
        scaled_features = scaler.transform(raw_features)
        
        with torch.no_grad():
            pred_lst = float(model(torch.tensor(scaled_features, dtype=torch.float32)).item())

        print(f"🔹 Scenario: {name}")
        print(f"   Inputs: NDVI={params['NDVI']}, Albedo={params['Albedo']}, BuildDensity={params['Building_Density']}, AirTemp={params['Air_Temp']}°C, Hum={params['Humidity']}%")
        print(f"   Predicted LST: {pred_lst:.4f}°C")
        
        # Analyze sanity
        lst_air_delta = pred_lst - params["Air_Temp"]
        print(f"   LST - Air_Temp Delta: {lst_air_delta:.4f}°C")
        
        # Auditing logic
        flaw_detected = False
        reasons = []

        if pred_lst < -273.15:
            flaw_detected = True
            reasons.append("Below absolute zero (impossible physics!).")
        elif pred_lst < 0.0 and params["Air_Temp"] > 0.0:
            flaw_detected = True
            reasons.append("Model outputs sub-zero LST while air temperature is comfortably warm (impossible thermodynamic behavior).")
        
        # Check if the output collapses to training mean LST (~45.3°C) under wildly different air temperatures
        if abs(pred_lst - 45.33) < 1.0 and abs(params["Air_Temp"] - 45.33) > 15.0:
            flaw_detected = True
            reasons.append("Trivial mean collapse. Model ignores the temperature features and predicts the global average.")

        if flaw_detected:
            print(f"AUDIT FLAW DETECTED: {', '.join(reasons)}")
        else:
            print("PASS: Biophysical output checks out as plausible.")
        print("--------------------------------------------------")

    print("\nAudit sequence finished.")

if __name__ == "__main__":
    run_adversarial_audit()
