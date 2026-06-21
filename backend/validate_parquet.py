import os
import sys
import pandas as pd
import numpy as np
import h3
import config

def validate_dataset():
    print("==================================================")
    print("Initiating Production-Grade Data Quality Gate...")
    print("==================================================")
    
    path = config.OUTPUT_PARQUET_PATH
    if not os.path.exists(path):
        print(f"CRITICAL ERROR: Output file {path} does not exist!")
        sys.exit(1)
        
    # Read Parquet DataFrame
    df = pd.read_parquet(path)
    
    expected_cols = ["H3_Index_ID", "LST", "NDVI", "Albedo", "Building_Density", "Air_Temp", "Humidity"]
    
    print(f"File Size: {os.path.getsize(path) / 1024:.2f} KB")
    print(f"Total Rows (Hexagonal Cells): {len(df)}")
    
    # 1. Structural Validation
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        print(f"CRITICAL ERROR: Missing expected columns: {missing_cols}")
        sys.exit(1)
    print(" PASS: Column structure is correct.")
        
    # 2. H3 Token Integrity Check
    sample_indices = df["H3_Index_ID"].dropna().head(10).tolist()
    
    def check_h3_valid(idx):
        if hasattr(h3, 'is_valid_cell'):
            return h3.is_valid_cell(idx)
        elif hasattr(h3, 'h3_is_valid'):
            return h3.h3_is_valid(idx)
        return False
        
    if not all(check_h3_valid(idx) for idx in sample_indices):
        print("CRITICAL ERROR: 'H3_Index_ID' contains invalid Uber H3 index tokens!")
        sys.exit(1)
    print("PASS: H3 spatial tokens are healthy and valid.")

    # 3. Null Value Audit
    nulls = df.isnull().sum()
    print("\n Null Value Audit:")
    for col, n_count in nulls.items():
        print(f"  - {col}: {n_count} nulls ({n_count/len(df)*100:.2f}%)")
        
    if nulls.sum() > 0:
        print(" WARNING: Dataset contains null records. Check cloud masking filters.")
    else:
        print(" PASS: Clean tabular dataset with 0 null values.")
        
    # 4. Numeric Range & Spatial Join Flatline Validations
    print("\nBiophysical Range & Scaling Audit:")
    failure_flag = False
    
    # LST Validation
    lst_min, lst_max = df["LST"].min(), df["LST"].max()
    if lst_min < 0 or lst_max > 65.0:
        print(f"  - LST range out of bounds: [{lst_min:.2f}, {lst_max:.2f}] °C")
        if df["LST"].mean() > 1000.0:
            print("    🚨 CRITICAL: Digital Numbers detected. Radiometric conversion multipliers were skipped!")
        failure_flag = True
    else:
        print(f"  - LST range: [{lst_min:.2f}, {lst_max:.2f}] °C (PASS)")
        
    # NDVI Validation
    if df["NDVI"].min() < -1.0 or df["NDVI"].max() > 1.0 or df["NDVI"].std() == 0:
        print(f"  -  NDVI anomaly detected: Range [{df['NDVI'].min():.2f}, {df['NDVI'].max():.2f}], StdDev: {df['NDVI'].std():.2f}")
        failure_flag = True
    else:
        print(f"  - NDVI range: [{df['NDVI'].min():.2f}, {df['NDVI'].max():.2f}] (PASS)")
        
    # Albedo Validation
    if df["Albedo"].min() < 0.0 or df["Albedo"].max() > 1.0 or df["Albedo"].std() == 0:
        print(f"  -  Albedo anomaly detected: Range [{df['Albedo'].min():.2f}, {df['Albedo'].max():.2f}], StdDev: {df['Albedo'].std():.2f}")
        failure_flag = True
    else:
        print(f"  - Albedo range: [{df['Albedo'].min():.2f}, {df['Albedo'].max():.2f}] (PASS)")
        
    # Building Density Spatial Join Check
    if df["Building_Density"].min() < 0.0 or df["Building_Density"].max() > 1.0 or df["Building_Density"].std() == 0:
        print(f"  -  Building_Density anomaly: Range [{df['Building_Density'].min():.2f}, {df['Building_Density'].max():.2f}]. If StdDev is 0, spatial joins flatlined!")
        failure_flag = True
    else:
        print(f"  - Building_Density range: [{df['Building_Density'].min():.2f}, {df['Building_Density'].max():.2f}] (PASS)")
        
    if failure_flag:
        print("\n DATA QUALITY GATE FAILED. Stopping execution pipeline.")
        sys.exit(1)
        
    # Print Descriptive Statistics
    print("\n Summary Statistics Matrix:")
    print(df.describe().T[["mean", "std", "min", "max"]])
    
    print("\n Dataset Sample View (First 5 rows):")
    print(df.head(5).to_string(index=False))
    
    print("==================================================")
    print(" DATA GATEWAY PASSED: Ready for AI/ML Modeling Phase.")
    print("==================================================")
    sys.exit(0)

if __name__ == "__main__":
    validate_dataset()