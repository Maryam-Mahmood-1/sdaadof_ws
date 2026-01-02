import numpy as np

class CBF_SuperEllipsoid:
    """
    Module 4: Control Barrier Function (Safety).
    Implements Super-Ellipsoid Safe Set (Eq 24-26).
    """
    def __init__(self, center, lengths, power_n=4):
        self.xc = np.array(center)
        self.n = int(power_n)
        
        # Matrix A setup (same as before)
        self.dims = np.array(lengths)
        self.A_diag = 1.0 / (self.dims ** (2 * self.n))
        self.A = np.diag(self.A_diag)

        
        self.k1 = 100.0       
        self.k0 = 90.0       

    def get_h(self, x):
        """ Barrier Value h(x) (Eq 24) """
        diff = x - self.xc
        abs_diff_n = np.abs(diff) ** self.n
        term = abs_diff_n.T @ self.A @ abs_diff_n
        return 1.0 - term

    def get_h_dot(self, x, dx):
        """ Time Derivative h_dot(x) (Eq 25) """
        diff = x - self.xc
        sign_diff = np.sign(diff)
        
        # Chain rule vector term (X_tilde in paper text)
        pow_n_1 = np.abs(diff) ** (self.n - 1)
        X_tilde = sign_diff * self.n * pow_n_1 * dx 
        
        term_abs_n = np.abs(diff) ** self.n
        
        # h_dot = -2 * [|x-xc|^n]^T * A * X_tilde
        h_dot = -2 * term_abs_n.T @ self.A @ X_tilde
        return h_dot, X_tilde

    def get_constraints(self, x, dx, a_des):
        """
        Returns Linear Constraints for QP: A_cbf * mu >= b_cbf
        Constraint: h_ddot + k1*h_dot + k0*h >= 0
        Since h_ddot is linear in mu (via x_ddot = a_des + mu),
        we return the terms to isolate mu.
        """
        h_val = self.get_h(x)
        h_dot, X_tilde = self.get_h_dot(x, dx)
        
        # --- Calculate h_ddot terms (Eq 26) ---
        diff = x - self.xc
        sign_diff = np.sign(diff)
        term_abs_n = np.abs(diff) ** self.n
        
        # 1. Term from derivative of X_tilde (velocity part)
        pow_n_2 = np.abs(diff) ** (self.n - 2)
        dX_tilde_drift = self.n * (self.n - 1) * pow_n_2 * (dx * dx)
        
        # 2. Term from derivative of X_tilde (acceleration coefficient)
        pow_n_1 = np.abs(diff) ** (self.n - 1)
        dX_tilde_acc_coeff = self.n * sign_diff * pow_n_1
        
        # 3. Combine to form h_ddot structure
        # h_ddot = const_terms + (Gradient @ x_ddot)
        
        term1 = -2 * X_tilde.T @ self.A @ X_tilde
        h_ddot_drift = term1 - 2 * term_abs_n.T @ self.A @ dX_tilde_drift
        
        # Gradient vector w.r.t x_ddot
        grad_x_ddot = -2 * term_abs_n.T @ self.A @ np.diag(dX_tilde_acc_coeff)
        
        # --- Formulate QP Constraint ---
        # h_ddot = h_ddot_drift + grad_x_ddot @ (a_des + mu)
        # Constraint: (h_ddot_drift + grad @ a_des + grad @ mu) + k1*h_dot + k0*h >= 0
        # Rearrange:  grad @ mu >= -h_ddot_drift - grad @ a_des - k1*h_dot - k0*h
        
        L_cbf = grad_x_ddot  # The 'A' matrix for this constraint
        boundary_val = -h_ddot_drift - (grad_x_ddot @ a_des) - self.k1*h_dot - self.k0*h_val
        
        return L_cbf, boundary_val