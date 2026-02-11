import numpy as np
import logging

# Setup logger for geometry module
logger = logging.getLogger(__name__)

class GeometricExtrapolator:
    """
    Implements geometric extrapolation for state matrices W_t ∈ R^{m×n}.

    Computes trajectory metrics (tau and kappa_eff) and performs extrapolation
    based on dynamic subspace estimated via PCA/SVD over velocities.
    
    Extrapolation combines velocity, acceleration and orthogonal escape components
    with adaptive coefficients based on geometric metrics.
    """

    def __init__(
        self,
        beta0=None,
        gamma0=None,
        eta0=None,
        k_subspace=5,
        ell_orth=3,
        eps=1e-8,
    ):
        """
        Args:
            beta0: base gain for linear term.
            alpha_beta: relative amplitude of beta adaptation.
            lambda_tau: beta sensitivity to inefficiency τ.
            gamma0: maximum gain for acceleration correction.
            c_gamma: gamma sensitivity to curvature κ_eff.
            eta0: base gain for orthogonal escape term.
            k_subspace: dimension of dynamic subspace U.
            ell_orth: number of secondary modes used for spectral escape.
            eps: numerical regularization.
        """
        self.beta0 = beta0
        self.gamma0 = gamma0
        self.eta0 = eta0
        self.k_subspace = k_subspace
        self.ell_orth = ell_orth
        self.eps = eps

    # Utility methods

    def _fro_norm(self, A: np.ndarray) -> float:
        return float(np.linalg.norm(A))

    def _flatten(self, A: np.ndarray) -> np.ndarray:
        """Converts matrix to one-dimensional vector."""
        return A.reshape(-1)

    def _unflatten(self, v: np.ndarray, shape) -> np.ndarray:
        """Reconstructs matrix from one-dimensional vector."""
        return v.reshape(shape)

    # Geometric metrics computation

    def compute_tau(self, W_hist: np.ndarray) -> float:
        """
        Computes trajectory elongation factor (ratio between total path and direct distance).
        """
        S = W_hist.shape[0]
        if S < 2:
            return 1.0

        # Calculate differences between consecutive states
        V = W_hist[1:] - W_hist[:-1]
        step_norms = np.linalg.norm(V, axis=(1, 2))
        S_path = float(np.sum(step_norms))

        delta = W_hist[-1] - W_hist[0]
        D = self._fro_norm(delta)

        tau = S_path / (D + self.eps)
        return float(tau)

    def compute_kappa_eff(self, W_hist: np.ndarray) -> float:
        """
        Computes effective trajectory curvature by measuring cumulative lateral deviation
        relative to global direction between initial and final states.
        """
        S = W_hist.shape[0]
        if S < 3:
            return 0.0

        delta = W_hist[-1] - W_hist[0]  # [m, n]
        D = self._fro_norm(delta)
        if D < self.eps:
            return 0.0

        U = delta / (D + self.eps)

        # Calculate incremental steps
        V = W_hist[1:] - W_hist[:-1]    # [S-1, m, n]

        # Project steps onto global direction
        alpha = np.tensordot(V, U, axes=([1, 2], [0, 1]))

        # Decompose into parallel and perpendicular components
        V_par = alpha[:, None, None] * U                  
        V_perp = V - V_par

        num = float(np.sum(np.linalg.norm(V_perp, axis=(1, 2))))
        den = float(np.sum(np.linalg.norm(V_par, axis=(1, 2)))) + self.eps

        kappa_eff = num / den
        return float(kappa_eff)
    
    def compute_metrics(self, W_hist: np.ndarray):
        # Validate minimum history
        if W_hist.shape[0] < 2:
             return {"tau": 1.0, "kappa_eff": 0.0}
        
        # Normalize 1D tensors for compatibility
        is_1d = (W_hist.ndim == 2) # [S, D]
        if is_1d:
            W_reshaped = W_hist[:, np.newaxis, :]
            tau = self.compute_tau(W_reshaped)
            kappa_eff = self.compute_kappa_eff(W_reshaped)
        else:
            tau = self.compute_tau(W_hist)
            kappa_eff = self.compute_kappa_eff(W_hist)
            
        return {"tau": tau, "kappa_eff": kappa_eff}

    # Dynamic subspace estimation and escape direction

    def _estimate_subspace_and_escape_dir(self, W_hist: np.ndarray):
        """
        Estimates dynamic subspace basis via PCA and orthogonal escape direction
        from secondary spectral modes. Returns None for degenerate cases.
        """
        S, m, n = W_hist.shape
        if S < 2:
            return None, None

        # Calculate local velocities
        V = W_hist[1:] - W_hist[:-1]
        T = S - 1
        D = m * n

        # Flatten for PCA analysis
        V_flat = V.reshape(T, D)

        # Data centering
        mean_vel = V_flat.mean(axis=0, keepdims=True)      # [1, D]
        V_centered = V_flat - mean_vel                     # [T, D]

        # Degeneracy check
        if np.linalg.norm(V_centered) < self.eps:
            return None, None

        # Singular value decomposition with safety checks
        try:
            # Check matrix size to prevent excessive computation
            matrix_size = V_centered.shape[0] * V_centered.shape[1]
            if matrix_size > 1e8:  # 100M elements
                logger.warning(f"⚠️ Large matrix for SVD: {V_centered.shape}, may be slow")
            
            U_svd, Svals, Vt = np.linalg.svd(V_centered, full_matrices=False)
            Q = Vt.T   # [D, R], R = effective rank (<= min(T, D))
        except Exception as e:
            logger.error(f"❌ SVD failed: {e}")
            return None, None

        # Extract principal subspace
        k_eff = min(self.k_subspace, Q.shape[1])
        Q_k = Q[:, :k_eff] if k_eff > 0 else None

        # Construct escape direction
        start = k_eff
        end = min(k_eff + self.ell_orth, Q.shape[1])

        v_orth_dir = None
        if start < end:
            # Weighted combination by singular values
            coeffs = Svals[start:end]
            Q_sec = Q[:, start:end]
            v_orth_flat = Q_sec @ coeffs  # [D]

            norm_orth = np.linalg.norm(v_orth_flat)
            if norm_orth > self.eps:
                v_orth_dir = self._unflatten(
                    v_orth_flat / (norm_orth + self.eps),
                    (m, n),
                )

        return Q_k, v_orth_dir

    def _project_in_subspace(self, X: np.ndarray, Q_k: np.ndarray) -> np.ndarray:
        """
        Projects matrix onto subspace specified by orthonormal basis Q_k.
        """
        if Q_k is None:
            return X

        m, n = X.shape
        x_flat = self._flatten(X)
        # Orthogonal projection
        x_proj_flat = Q_k @ (Q_k.T @ x_flat)
        return self._unflatten(x_proj_flat, (m, n))

    # Adaptive coefficients computation

    def beta(self, tau: float) -> float:
        """
        Computes velocity coefficient inversely proportional to elongation factor.
        """
        return float(self.beta0 / (tau + self.eps))

    def gamma(self, kappa: float) -> float:
        """
        Computes acceleration coefficient inversely proportional to curvature.
        """
        return float(self.gamma0 / (1.0 + kappa))

    def eta(self, tau: float, kappa: float) -> float:
        """
        Computes escape coefficient proportional to product of tau and kappa.
        """
        return float(self.eta0 * tau * kappa)
    

    # State extrapolation

    def extrapolate(self, W_hist: np.ndarray):
        """
        W_hist: 
           - [S, m, n] for weights (Matrices)
           - [S, d] for bias (Vectors)
        """
        # Dimensionality normalization
        original_shape = W_hist.shape[1:]
        is_1d = (len(original_shape) == 1)
        
        # Add heartbeat logging
        logger.debug(f"🔍 Extrapolating: shape={W_hist.shape}, is_1d={is_1d}")
        
        try:
            logger.debug("   Computing metrics...")
            metrics = self.compute_metrics(W_hist)
            tau = metrics["tau"]
            kappa = metrics["kappa_eff"]
            logger.debug(f"   ✅ Metrics: tau={tau:.4f}, kappa={kappa:.4f}")
        except Exception as e:
            logger.error(f"❌ Failed to compute metrics in extrapolate: {e}")
            raise
        
        W_last = W_hist[-1]
        
        # Calculate average velocity and acceleration
        V = W_hist[1:] - W_hist[:-1]
        v_mean = V.mean(axis=0)
        
        # Acceleration when history is sufficient
        if W_hist.shape[0] >= 3:
            A = W_hist[2:] - 2 * W_hist[1:-1] + W_hist[:-2]
            a_mean = A.mean(axis=0)
        else:
            a_mean = np.zeros_like(W_last)

        # Subspace estimation for 2D tensors
        Q_k, v_orth_dir = None, None
        if not is_1d:
            try:
                logger.debug("   Estimating subspace...")
                Q_k, v_orth_dir = self._estimate_subspace_and_escape_dir(W_hist)
                logger.debug("   ✅ Subspace estimated")
            except Exception as e:
                logger.warning(f"⚠️ Subspace estimation failed (using fallback): {e}")
                Q_k, v_orth_dir = None, None
        
        # Project onto subspace when available
        v_U = self._project_in_subspace(v_mean, Q_k) if not is_1d else v_mean
        a_U = self._project_in_subspace(a_mean, Q_k) if not is_1d else a_mean
        
        # Vector normalization
        v_norm = self._fro_norm(v_U)
        a_norm = self._fro_norm(a_U)
        
        v_hat = v_U / (v_norm + self.eps) if v_norm > self.eps else np.zeros_like(W_last)
        a_hat = a_U / (a_norm + self.eps) if a_norm > self.eps else np.zeros_like(W_last)
        
        # Orthogonal escape direction
        v_perp_hat = np.zeros_like(W_last)
        if v_orth_dir is not None:
            v_perp_hat = v_orth_dir
        elif not is_1d:
            delta = W_hist[-1] - W_hist[0]
            delta_norm = self._fro_norm(delta)
             
        # Calculate adaptive coefficients
        beta_val = self.beta(tau)
        gamma_val = self.gamma(kappa)
        eta_val = self.eta(tau, kappa)
        
        print("beta_val: ", beta_val)
        print("gamma_val: ", gamma_val)
        print("eta_val: ", eta_val)
        v_U = self._project_in_subspace(v_mean, Q_k) if not is_1d else v_mean
        a_U = self._project_in_subspace(a_mean, Q_k) if not is_1d else a_mean

        # Extrapolation components
        step_linear = beta_val * v_U
        step_curv = gamma_val * a_U 
        step_escape = eta_val * v_perp_hat
        
        W_new = W_last + step_linear + step_curv + step_escape
        
        metrics["delta_norm"] = self._fro_norm(W_hist[-1] - W_hist[0])
        
        return W_new, metrics
