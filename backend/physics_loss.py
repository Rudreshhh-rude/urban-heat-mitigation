import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """
    Custom Physics-Informed Neural Network (PINN) Loss Module for Urban Heat Mitigation.
    
    This loss function combines empirical data loss (MSE of predicted vs observed Land 
    Surface Temperature) with a thermodynamic energy balance penalty:
    
        Rn + Af = H + LE + G  ==>  Residual = Rn - (H + LE + G) (assuming Af = 0 proxy)
        
    It handles:
    1. Gradient Flow Breakdown: Using Softplus or Leaky Clamp instead of max(0, NDVI)
       to ensure gradients propagate for negative NDVI (concrete, water, etc.).
    2. Loss Scale Balancing: Allowing the physics residual to be normalized by S_solar
       (bringing it to a 0-1 scale) or using a raw scaling factor to balance with MSE.
    """
    def __init__(
        self, 
        lambda_physics=1e-5, 
        S_solar=500.0, 
        C_transpiration=400.0, 
        h_0=10.0, 
        h_1=20.0,
        gradient_mode='softplus',  # 'softplus', 'leaky_clamp', or 'strict'
        normalize_residual=False    # If True, scales residual by S_solar to balance loss magnitudes
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.S_solar = S_solar
        self.C_transpiration = C_transpiration
        self.h_0 = h_0
        self.h_1 = h_1
        self.gradient_mode = gradient_mode
        self.normalize_residual = normalize_residual
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_lst, true_lst, raw_features):
        """
        Calculates the combined physics-informed loss.
        
        Parameters:
        -----------
        pred_lst : torch.Tensor
            Predicted Land Surface Temperature (LST) in Celsius, shape (batch_size,) or (batch_size, 1).
        true_lst : torch.Tensor
            Observed satellite LST in Celsius, shape (batch_size,) or (batch_size, 1).
        raw_features : torch.Tensor
            Unscaled, raw input features, shape (batch_size, num_features).
            Expected column ordering:
                0: NDVI
                1: Albedo
                2: Building_Density
                3: Air_Temp
                4: Humidity (unused in physics equations)
        """
        # Ensure outputs and targets are flattened 1D tensors
        pred_lst = pred_lst.squeeze()
        true_lst = true_lst.squeeze()
        
        # 1. Data Loss (MSE of Land Surface Temperature)
        data_loss = self.mse_loss(pred_lst, true_lst)
        
        # 2. Extract Raw Features
        ndvi = raw_features[:, 0]
        albedo = raw_features[:, 1]
        building_density = raw_features[:, 2]
        air_temp = raw_features[:, 3]
        
        # 3. Calculate Flux Proxies
        
        # 3.1 Net Radiation (Rn)
        r_n = self.S_solar * (1.0 - albedo)
        
        # 3.2 Latent Heat Flux (LE)
        # Handle gradient flow breakdown for negative NDVI values
        if self.gradient_mode == 'softplus':
            # softplus(x) is a smooth approximation of max(0, x)
            ndvi_effective = F.softplus(ndvi, beta=10.0)
        elif self.gradient_mode == 'leaky_clamp':
            # leaky clamp retains a small gradient (0.01) for negative NDVI values
            ndvi_effective = torch.where(ndvi > 0.0, ndvi, 0.01 * ndvi)
        else:
            # strict clamp (zero gradients for ndvi < 0)
            ndvi_effective = torch.clamp(ndvi, min=0.0)
            
        le = self.C_transpiration * ndvi_effective
        
        # 3.3 Sensible Heat Flux (H)
        h = (self.h_0 + self.h_1 * building_density) * (pred_lst - air_temp)
        
        # 3.4 Ground Storage Flux (G)
        g = 0.15 * r_n
        
        # 4. Calculate Physics Residual
        residual = r_n - (h + le + g)
        
        # 5. Physics Loss Calculation (with scale balancing options)
        if self.normalize_residual:
            # Normalize residual by S_solar to make it dimensionless and aligned with MSE scale
            residual_scaled = residual / self.S_solar
            physics_loss = torch.mean(residual_scaled ** 2)
        else:
            # Raw physics loss (requires very small lambda_physics, e.g. 1e-5 or 1e-6)
            physics_loss = torch.mean(residual ** 2)
            
        # 6. Combined Loss
        total_loss = data_loss + (self.lambda_physics * physics_loss)
        
        return total_loss, data_loss, physics_loss
