import numpy as np

class CBF_SuperEllipsoid:
    """
    Implements the 'Virtual Cage' Super-Ellipsoid Safety Constraint.
    
    Paper Ref: "Safety Compliant Control for Robotic Manipulator..."
    
    Mathematical Definition (Eq. 24):
    h(x) = 1 - [|x - x_c|^n]^T A [|x - x_c|^n]
    
    where A is the diagonal matrix of inverse dimensions (1/length^2n, etc).
    """
    def __init__(self, center, lengths, power_n=4, k_pos=10.0, k_vel=5.0):
        """
        Args:
            center (list): x_c from Eq. 24
            lengths (list): Dimensions defining Matrix A in Eq. 21
            power_n (int): n in Eq. 22 (must be even)
            k_pos, k_vel: Tuning gains K corresponding to Eq. 19
        """
        self.center = np.array(center)
        self.radii = np.array(lengths)
        self.n = power_n
        self.kp = k_pos
        self.kv = k_vel

    def get_constraints(self, x, dx, x_ref_ddot):
        """
        Calculates the linear constraint A*mu <= b for the QP.
        
        Constraint (ECBF Eq. 19):
        L_f^2 h(x) + L_g L_f h(x) Γ + K η >= 0
        
        In our acceleration-controlled form (since Γ controls x_ddot):
        h_ddot(x) + K_v h_dot(x) + K_p h(x) >= 0
        
        Returns:
            A (1x3), b (1x1)
        """
        # 1. Normalized Position vector term: w = (x - x_c) / radii
        w = (x - self.center) / self.radii
        
        # ---------------------------------------------------------
        # Eq. 24: h(x) = 1 - [|x - x_c|^n]^T A [|x - x_c|^n]
        # ---------------------------------------------------------
        # Note: We implement this element-wise first
        term_pow = w ** self.n
        h = 1.0 - np.sum(term_pow)
        
        # ---------------------------------------------------------
        # Eq. 25: h_dot(x) (First Derivative)
        # ḣ(x) = -2 [|x - x_c|^n]^T A [sign(x-x_c) ◦ n(x-x_c)^(n-1) ◦ ẋ]
        # ---------------------------------------------------------
        # Simplified Implementation:
        # ∂h/∂x_i = -n * (w_i)^(n-1) * (1/r_i)
        grad_h = -self.n * (w ** (self.n - 1)) / self.radii
        
        # ḣ = ∇h ⋅ ẋ
        h_dot = np.dot(grad_h, dx)
        
        # ---------------------------------------------------------
        # Eq. 26: h_ddot(x) (Second Derivative Terms)
        # ḧ(x) = L_f^2 h(x) + L_g L_f h(x) μ
        # ---------------------------------------------------------
        
        # Calculate time derivative of gradient: d(∇h)/dt
        # d/dt( w^(n-1) ) = (n-1) * w^(n-2) * ẇ
        # ẇ = ẋ / r
        w_dot = dx / self.radii
        d_grad_dt = -self.n * (self.n - 1) * (w ** (self.n - 2)) * w_dot / self.radii
        
        # The "Drift" part: L_f^2 h(x) corresponds to terms independent of input μ
        # ḧ_drift = (d(∇h)/dt ⋅ ẋ) + (∇h ⋅ ẍ_ref)
        h_ddot_drift = np.dot(d_grad_dt, dx)
        
        # ---------------------------------------------------------
        # Eq. 19: ECBF Inequality
        # ḧ + K η >= 0
        # (L_f^2 h + L_g L_f h μ) + (K_v ḣ + K_p h) >= 0
        # ---------------------------------------------------------
        
        # Lg_h is the term multiplying μ (Control Influence)
        Lg_h = grad_h 
        
        # K η term (PD-like behavior for boundary)
        barrier_term = self.kv * h_dot + self.kp * h
        
        # Rearranging for QP (A μ <= b):
        # ∇h ⋅ μ >= - (ḧ_drift + ∇h ⋅ ẍ_ref + K_v ḣ + K_p h)
        # -∇h ⋅ μ <= (ḧ_drift + ∇h ⋅ ẍ_ref + K_v ḣ + K_p h)
        
        upper_bound_accel = -(h_ddot_drift + np.dot(grad_h, x_ref_ddot) + barrier_term)
        
        A_cbf = -Lg_h.reshape(1, 3)
        b_cbf = -upper_bound_accel.reshape(1, 1)
        
        return A_cbf, b_cbf