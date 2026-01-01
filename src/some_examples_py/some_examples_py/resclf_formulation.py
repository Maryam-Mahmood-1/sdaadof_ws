import numpy as np
from scipy.linalg import solve_continuous_are

class RESCLF_Formulation:
    """
    Module 2: RESCLF Stability Mathematics.
    Implements Eq (11) through (15).
    """
    def __init__(self, dim=3):
        self.dim = dim
        
        # Matrices F and G (Eq 12)
        # F = [0 I; 0 0], G = [0; I]
        self.F = np.zeros((2*dim, 2*dim))
        self.F[:dim, dim:] = np.eye(dim)
        self.G = np.zeros((2*dim, dim))
        self.G[dim:, :] = np.eye(dim)
        
        # Weights for ARE (Eq 13)
        self.Q = np.eye(2*dim) * 1000.0 
        self.R = np.eye(dim) * 0.1
        
        # Solve ARE: F.T*P + P*F - P*G*R_inv*G.T*P + Q = 0
        self.P = solve_continuous_are(self.F, self.G, self.Q, self.R)
        
        # Calculate Gamma for Stability Condition (Eq 15)
        # gamma = min_eig(Q) / max_eig(P)
        min_Q = np.min(np.linalg.eigvals(self.Q).real)
        max_P = np.max(np.linalg.eigvals(self.P).real)
        self.gamma = 1.8 * (min_Q / max_P)

    def get_qp_constraints(self, e, de):
        """
        Calculates Lie Derivatives and V for Eq (15).
        Returns LfV, LgV, V_val, gamma
        """
        eta = np.hstack([e, de]) # Error state vector [e; e_dot]
        
        # Lyapunov Function V (Eq 14 context)
        V_val = eta.T @ self.P @ eta
        
        # Gradient dV/d_eta = 2 * eta.T * P
        dV_deta = 2 * eta.T @ self.P
        
        # Lie Derivatives (Eq 14)
        LfV = dV_deta @ (self.F @ eta)
        LgV = dV_deta @ self.G
        
        return LfV, LgV, V_val, self.gamma