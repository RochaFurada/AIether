import torch
import torch.nn as nn
import numpy as np
from aiether.utils import GeometricExtrapolator, logger_setup

logger = logger_setup()

class GrowthManager:
    """
    Manages initialization of new layers through geometric extrapolation
    applied individually to each parameter tensor.
    """
    def __init__(self, beta0=None, gamma0=None, eta0=None, k_subspace=None, ell_orth=None):
        # Initialize geometric extrapolator with provided parameters
        self.geometric_extrapolator = GeometricExtrapolator(
            k_subspace=k_subspace,  
            ell_orth=ell_orth,
            eta0=eta0,
            beta0=beta0,
            gamma0=gamma0,
            eps=1e-8,
        )
        # Dictionary to store statistics from last initialization
        self.last_strategy_stats = {}

    def initialize_new_layer_from_history(self, new_layer: nn.Module, param_history: dict, strategy: str):
        """
        Initializes `new_layer` using parameter history from previous layer.

        Args:
            new_layer (nn.Module): The new layer to be initialized.
            param_history (dict): Dictionary {param_name: list[np.ndarray]}. 
                                  Each list contains the temporal timeline of tensor snapshots.
            strategy (str): Strategy name (for logging/control only).
        """
        logger.info(f"🔧 Initializing new layer via '{strategy}' (Tensor-wise Extrapolation)")
        
        self.last_strategy_stats = {}
        
        # Process each parameter of the new layer
        with torch.no_grad():
            for name, param in new_layer.named_parameters():
                # Check history availability for parameter
                if name not in param_history:
                    logger.warning(f"⚠️ History not found for '{name}'. Copying last available state or default initialization.")
                    continue
                
                # Retrieve parameter temporal history
                history_list = param_history[name]
                
                if len(history_list) < 2:
                    logger.warning(f"⚠️ Insufficient history for '{name}' (len={len(history_list)}). Copying last frame.")
                    param.data.copy_(torch.from_numpy(history_list[-1]).to(param.device))
                    continue

                # Stack temporal snapshots
                W_hist = np.stack(history_list, axis=0)
                
                # Apply geometric extrapolation
                W_new_np, metrics = self.geometric_extrapolator.extrapolate(W_hist)
                
                # Convert result to PyTorch tensor
                W_new_tensor = torch.from_numpy(W_new_np).to(dtype=param.dtype, device=param.device)
                
                # Update parameter with extrapolated value
                param.data.copy_(W_new_tensor)
                
                # Register extrapolation metrics
                self.last_strategy_stats[name] = {
                    "shape": str(W_hist.shape),
                    "tau": f"{metrics.get('tau', 0):.4f}",
                    "kappa": f"{metrics.get('kappa_eff', 0):.4f}",
                    "delta_norm": f"{metrics.get('delta_norm', 0):.6f}"
                }

        logger.info("✅ Tensor-wise initialization completed.")
