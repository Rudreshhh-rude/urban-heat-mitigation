import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from physics_loss import PhysicsInformedLoss

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define PyTorch Dataset
class UrbanHeatDataset(Dataset):
    """
    Dataset that returns:
    - X_scaled: Scaled features for neural network training
    - X_raw: Raw, unscaled physical features for physics loss calculations
    - y: Target values (observed LST)
    """
    def __init__(self, X_scaled, X_raw, y):
        self.X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
        self.X_raw = torch.tensor(X_raw, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_scaled[idx], self.X_raw[idx], self.y[idx]

# Define MLP Model
class UrbanThermalMLP(nn.Module):
    """
    A simple Multilayer Perceptron for Land Surface Temperature prediction.
    Inputs (5 dimensions): NDVI, Albedo, Building_Density, Air_Temp, Humidity
    Output (1 dimension): LST
    """
    def __init__(self, input_dim=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Training Epoch
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_data_loss = 0.0
    running_phys_loss = 0.0
    total_samples = 0

    for X_scaled, X_raw, y in loader:
        X_scaled, X_raw, y = X_scaled.to(device), X_raw.to(device), y.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_scaled)
        
        loss, data_loss, phys_loss = criterion(predictions, y, X_raw)
        loss.backward()
        optimizer.step()
        
        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        running_data_loss += data_loss.item() * batch_size
        running_phys_loss += phys_loss.item() * batch_size
        total_samples += batch_size

    return (
        running_loss / total_samples,
        running_data_loss / total_samples,
        running_phys_loss / total_samples
    )

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_data_loss = 0.0
    running_phys_loss = 0.0
    total_samples = 0
    
    all_preds = []
    all_targets = []
    all_raw_features = []

    with torch.no_grad():
        for X_scaled, X_raw, y in loader:
            X_scaled, X_raw, y = X_scaled.to(device), X_raw.to(device), y.to(device)
            
            predictions = model(X_scaled)
            loss, data_loss, phys_loss = criterion(predictions, y, X_raw)
            
            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            running_data_loss += data_loss.item() * batch_size
            running_phys_loss += phys_loss.item() * batch_size
            total_samples += batch_size
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_raw_features.append(X_raw.cpu().numpy())

    all_preds = np.vstack(all_preds).squeeze()
    all_targets = np.vstack(all_targets).squeeze()
    all_raw_features = np.vstack(all_raw_features)
    
    # Calculate physical residual statistics manually on validation set
    # Rn = S_solar * (1.0 - Albedo)
    # LE = C_transpiration * max(0, NDVI)
    # H = (h_0 + h_1 * Building_Density) * (pred_lst - Air_Temp)
    # G = 0.15 * Rn
    # Residual = Rn - (H + LE + G)
    ndvi = all_raw_features[:, 0]
    albedo = all_raw_features[:, 1]
    building_density = all_raw_features[:, 2]
    air_temp = all_raw_features[:, 3]
    
    r_n = criterion.S_solar * (1.0 - albedo)
    
    # Use exact formula used during evaluation (strict clamp for physical residual reporting)
    ndvi_clamped = np.maximum(0.0, ndvi)
    le = criterion.C_transpiration * ndvi_clamped
    h = (criterion.h_0 + criterion.h_1 * building_density) * (all_preds - air_temp)
    g = 0.15 * r_n
    
    residuals = r_n - (h + le + g)
    mean_residual = np.mean(residuals)
    mean_abs_residual = np.mean(np.abs(residuals))
    rmse = np.sqrt(running_data_loss / total_samples)
    mae = np.mean(np.abs(all_preds - all_targets))

    return {
        "val_total_loss": running_loss / total_samples,
        "val_data_loss_mse": running_data_loss / total_samples,
        "val_phys_loss": running_phys_loss / total_samples,
        "val_rmse": rmse,
        "val_mae": mae,
        "mean_residual": mean_residual,
        "mean_abs_residual": mean_abs_residual,
        "residuals": residuals
    }

def run_experiment(lambda_val, data_loaders, device, args):
    print(f"\n--- Training with lambda_physics = {lambda_val} ---")
    
    model = UrbanThermalMLP(input_dim=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = PhysicsInformedLoss(
        lambda_physics=lambda_val,
        S_solar=args.s_solar,
        C_transpiration=args.c_trans,
        h_0=args.h0,
        h_1=args.h1,
        gradient_mode=args.grad_mode,
        normalize_residual=args.normalize_residual
    )
    
    train_loader, val_loader = data_loaders
    
    best_val_rmse = float('inf')
    best_metrics = None
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_data, train_phys = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Periodic validation
        if epoch % 10 == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            if val_metrics["val_rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["val_rmse"]
                best_metrics = val_metrics
                
            if epoch % 50 == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} (MSE: {train_data:.4f}, Phys: {train_phys:.4f}) | "
                      f"Val RMSE: {val_metrics['val_rmse']:.4f} | Mean Residual: {val_metrics['mean_residual']:.2f} W/m2")
                      
    return best_metrics

def main():
    parser = argparse.ArgumentParser(description="PINN Training and lambda_physics Experimentation")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--s-solar", type=float, default=500.0, help="Assumed baseline solar radiation constant (W/m^2)")
    parser.add_argument("--c-trans", type=float, default=400.0, help="Transpiration cooling coefficient (W/m^2)")
    parser.add_argument("--h0", type=float, default=10.0, help="Sensible heat baseline roughness coefficient")
    parser.add_argument("--h1", type=float, default=20.0, help="Sensible heat roughness scaling coefficient")
    parser.add_argument("--grad-mode", type=str, default="softplus", choices=["softplus", "leaky_clamp", "strict"],
                        help="Differentiable function to prevent dead gradients on negative NDVI")
    parser.add_argument("--normalize-residual", action="store_true", 
                        help="Normalize physics residuals by S_solar to balance data and physics loss scales")
    parser.add_argument("--final-lambda", type=float, default=None,
                        help="Override automatic lambda selection for final model training")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip the parameter sweep and directly train the final model using final-lambda")
    
    args = parser.parse_args()
    
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training pipeline on device: {device}")
    print(f"Gradient mode: {args.grad_mode} | Normalizing physics residuals: {args.normalize_residual}")
    
    # 1. Load Parquet Data
    parquet_path = config.OUTPUT_PARQUET_PATH
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at {parquet_path}. Please run pipeline.py first.")
        return
        
    df = pd.read_parquet(parquet_path)
    print(f"Loaded dataset with {len(df)} hexagonal grid cells.")
    
    # 2. Extract Features and Target
    feature_cols = ["NDVI", "Albedo", "Building_Density", "Air_Temp", "Humidity"]
    target_col = "LST"
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    if args.no_sweep:
        final_lambda = args.final_lambda if args.final_lambda is not None else 0.05
        print(f"\nSkipping parameter sweep. Training final model directly on full dataset with lambda = {final_lambda}...")
        
        # Preprocess full dataset for production model
        final_scaler = StandardScaler()
        final_scaler.fit(X)
        
        # MLOps Refinement: Upsample extreme thermal hotspots (LST > 50°C)
        hotspot_mask = y > 50.0
        if np.any(hotspot_mask):
            X_hotspots = X[hotspot_mask]
            y_hotspots = y[hotspot_mask]
            X_full = np.vstack([X] + [X_hotspots] * 35)
            y_full = np.concatenate([y] + [y_hotspots] * 35)
            print(f"Upsampling {len(X_hotspots)} hotspot cells (>50°C) by 35x for final training.")
        else:
            X_full = X
            y_full = y
            
        X_full_scaled = final_scaler.transform(X_full)
        full_dataset = UrbanHeatDataset(X_full_scaled, X_full, y_full)
        full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)
        
        final_model = UrbanThermalMLP(input_dim=5).to(device)
        optimizer = optim.Adam(final_model.parameters(), lr=args.lr)
        criterion = PhysicsInformedLoss(
            lambda_physics=final_lambda,
            S_solar=args.s_solar,
            C_transpiration=args.c_trans,
            h_0=args.h0,
            h_1=args.h1,
            gradient_mode=args.grad_mode,
            normalize_residual=args.normalize_residual
        )
        
        # Train final model on full dataset
        final_epochs = max(args.epochs, 150)
        for epoch in range(1, final_epochs + 1):
            train_loss, train_data, train_phys = train_epoch(final_model, full_loader, optimizer, criterion, device)
            if epoch % 50 == 0 or epoch == final_epochs:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} (MSE: {train_data:.4f}, Phys: {train_phys:.4f})")
                
        # Save model state dict
        torch.save(final_model.state_dict(), config.MODEL_PATH)
        print(f"Saved final model weights to '{config.MODEL_PATH}'")
        
        # Save scaler using pickle
        import pickle
        with open(config.SCALER_PATH, "wb") as f:
            pickle.dump(final_scaler, f)
        print(f"Saved scaler to '{config.SCALER_PATH}'")
        return
    
    # 3. Train/Validation Split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    
    # 5. Create PyTorch Dataloaders
    train_dataset = UrbanHeatDataset(X_train_scaled, X_train_raw, y_train)
    val_dataset = UrbanHeatDataset(X_val_scaled, X_val_raw, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    data_loaders = (train_loader, val_loader)
    
    # 6. Define lambda_physics values to sweep
    # If using normalized residual, values can be larger (e.g. 0.01 - 0.5)
    # If using unscaled raw residual, values should be tiny (e.g. 1e-6 - 1e-4)
    if args.normalize_residual:
        lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    else:
        lambdas = [0.0, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4]
        
    results = []
    
    print("\nStarting Physics Loss tuning loop...")
    print("====================================================================================================")
    
    for l_val in lambdas:
        best_val = run_experiment(l_val, data_loaders, device, args)
        
        results.append({
            "lambda": l_val,
            "val_rmse": best_val["val_rmse"],
            "val_mae": best_val["val_mae"],
            "val_phys_loss": best_val["val_phys_loss"],
            "mean_residual": best_val["mean_residual"],
            "mean_abs_residual": best_val["mean_abs_residual"]
        })
        
    # 7. Print summary table
    print("\n====================================================================================================")
    print("                                      EXPERIMENT SWEEP RESULTS                                      ")
    print("====================================================================================================")
    header = f"{'lambda_phys':<12} | {'Val RMSE (°C)':<15} | {'Val MAE (°C)':<14} | {'Val Phys Loss':<15} | {'Mean Residual (W/m²)':<22} | {'Mean Abs Residual':<18}"
    print(header)
    print("-" * len(header))
    for r in results:
        # Format lambda as string
        if r["lambda"] == 0.0:
            l_str = "0.0"
        elif r["lambda"] < 1e-3:
            l_str = f"{r['lambda']:.1e}"
        else:
            l_str = f"{r['lambda']:.4f}"
            
        print(f"{l_str:<12} | {r['val_rmse']:<15.4f} | {r['val_mae']:<14.4f} | {r['val_phys_loss']:<15.4e} | {r['mean_residual']:<22.2f} | {r['mean_abs_residual']:<18.2f}")
    print("====================================================================================================")
    
    # 8. Recommend the best configuration
    # The best configuration balances low LST prediction error (RMSE) with physical consistency (low absolute residual)
    # Let's write the recommendation to the console
    print("\nAnalysis & Recommendation:")
    print("  - When lambda_phys = 0.0 (Pure Data Baseline): The model ignores physical energy balance, leading to the highest physical residuals.")
    print("  - As lambda_phys increases: The physical residuals decrease, demonstrating that the network is learning to comply with energy balance constraints.")
    print("  - If lambda_phys becomes too high: LST accuracy (RMSE/MAE) will degrade because the model prioritizes the physics equation over observations.")
    
    best_phys_idx = np.argmin([r["mean_abs_residual"] for r in results])
    best_data_idx = np.argmin([r["val_rmse"] for r in results])
    
    print(f"\n  * Pure Data Best RMSE: {results[0]['val_rmse']:.4f} °C (Mean Abs Residual: {results[0]['mean_abs_residual']:.2f} W/m²)")
    print(f"  * Physics-Optimal (Minimum Absolute Residual) lambda_phys = {results[best_phys_idx]['lambda']}: RMSE: {results[best_phys_idx]['val_rmse']:.4f} °C (Mean Abs Residual: {results[best_phys_idx]['mean_abs_residual']:.2f} W/m²)")
    
    # Filter to find a lambda that decreases absolute residual substantially without increasing RMSE by more than 0.1C
    baseline_rmse = results[0]["val_rmse"]
    candidates = [r for r in results if r["val_rmse"] <= baseline_rmse + 0.1]
    # 8. Train and save final recommended model
    print("\nTraining final recommended model and saving weights/scaler...")
    if args.final_lambda is not None:
        final_lambda = args.final_lambda
    elif candidates:
        recommended = min(candidates, key=lambda x: x["mean_abs_residual"])
        final_lambda = recommended["lambda"]
    else:
        final_lambda = results[0]["lambda"]
        
    print(f"Final Lambda: {final_lambda}")
    
    # Preprocess full dataset for production model
    final_scaler = StandardScaler()
    final_scaler.fit(X)
    
    # MLOps Refinement: Upsample extreme thermal hotspots (LST > 50°C)
    hotspot_mask = y > 50.0
    if np.any(hotspot_mask):
        X_hotspots = X[hotspot_mask]
        y_hotspots = y[hotspot_mask]
        X_full = np.vstack([X] + [X_hotspots] * 35)
        y_full = np.concatenate([y] + [y_hotspots] * 35)
        print(f"Upsampling {len(X_hotspots)} hotspot cells (>50°C) by 35x for final training.")
    else:
        X_full = X
        y_full = y
        
    X_full_scaled = final_scaler.transform(X_full)
    full_dataset = UrbanHeatDataset(X_full_scaled, X_full, y_full)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)
    
    final_model = UrbanThermalMLP(input_dim=5).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=args.lr)
    criterion = PhysicsInformedLoss(
        lambda_physics=final_lambda,
        S_solar=args.s_solar,
        C_transpiration=args.c_trans,
        h_0=args.h0,
        h_1=args.h1,
        gradient_mode=args.grad_mode,
        normalize_residual=args.normalize_residual
    )
    
    # Train final model on full dataset
    # We train for 150 epochs to allow convergence on the full dataset
    final_epochs = max(args.epochs, 150)
    for epoch in range(1, final_epochs + 1):
        train_epoch(final_model, full_loader, optimizer, criterion, device)
        
    # Save model state dict
    torch.save(final_model.state_dict(), config.MODEL_PATH)
    print(f"Saved final model weights to '{config.MODEL_PATH}'")
    
    # Save scaler using pickle
    import pickle
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(final_scaler, f)
    print(f"Saved scaler to '{config.SCALER_PATH}'")

if __name__ == "__main__":
    main()

