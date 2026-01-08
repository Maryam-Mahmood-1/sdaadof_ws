import numpy as np
import cvxopt

def solve_optimization(LfV, LgV, V, gamma, torque_A=None, torque_b=None, cbf_A=None, cbf_b=None):
    """
    Solves the CLF-QP (Control Lyapunov Function based Quadratic Program).
    
    Mathematical Formulation:
    -------------------------
    Optimization Variable: μ (Auxiliary Control Input)
    
    Objective:
        min (μ)  ½ μᵀ H μ + fᵀ μ
        
    Subject to:
    
    1. Stability Constraint (CLF):
       LgV(x) μ ≤ -γ V(x) - LfV(x)
       
    2. Input Constraints (Torque Saturation):
       τ_min ≤ M(q)J†(u_ref + μ - J̇q̇) + n(q,q̇) ≤ τ_max
       (Rearranged into linear form A_torq μ ≤ b_torq)
       
    3. Safety Constraints (CBF - Optional):
       A_cbf μ ≤ b_cbf
    """
    
    # ---------------------------------------------------------
    # 1. Objective Function
    #    Minimize control deviation energy: ||μ||²
    #    Standard Form: min ½ xᵀ P x + qᵀ x
    # ---------------------------------------------------------
    # 3 variables for μ (x, y, z acceleration correction)
    n_vars = 3
    
    # H = 2I  (Factor of 2 because standard form is 1/2 xPx)
    H = 2.0 * np.eye(n_vars)
    
    # f = 0
    f = np.zeros(n_vars)

    # ---------------------------------------------------------
    # 2. Build Constraints (G x ≤ h)
    # ---------------------------------------------------------
    A_list = []
    b_list = []

    # --- Constraint A: RES-CLF Stability ---
    # Inequality: LgV ⋅ μ ≤ -γ V - LfV
    A_list.append(LgV) 
    b_list.append(np.array([[-gamma * V - LfV]]))

    # --- Constraint B: Torque Saturation ---
    # Form: A_torq ⋅ μ ≤ b_torq
    # (Matrices pre-calculated in main controller to save time here)
    if torque_A is not None and torque_b is not None:
        A_list.append(torque_A)
        b_list.append(torque_b)

    # --- Constraint C: Control Barrier Function (Safety) ---
    # Form: A_cbf ⋅ μ ≤ b_cbf
    if cbf_A is not None and cbf_b is not None:
        A_list.append(cbf_A)
        b_list.append(cbf_b)

    # 3. Stack Matrices for Solver
    if not A_list:
        return np.zeros(n_vars), True

    G_qp = np.vstack(A_list)
    h_qp = np.vstack(b_list)

    # 4. Solve using CVXOPT
    # Solves: min ½ xᵀ P x + qᵀ x  s.t.  G x ≤ h
    cvxopt.solvers.options['show_progress'] = False
    P_cvx = cvxopt.matrix(H)
    q_cvx = cvxopt.matrix(f)
    G_cvx = cvxopt.matrix(G_qp)
    h_cvx = cvxopt.matrix(h_qp)

    try:
        sol = cvxopt.solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        mu = np.array(sol['x']).flatten()
        return mu, True
        
    except ValueError:
        # Infeasible (Constraints conflict, e.g., Safety vs Limits)
        # Returns zero correction (Fall back to nominal control)
        return np.zeros(n_vars), False
    

    


""""Without CBF constraints - Basic CLF-QP solver."""
# import numpy as np
# import cvxopt

# def solve_optimization(LfV, LgV, V, gamma, torque_A=None, torque_b=None, cbf_A=None, cbf_b=None):
#     """
#     Solves the CLF-QP (Control Lyapunov Function based Quadratic Program).
    
#     Mathematical Formulation:
#     -------------------------
#     Optimization Variable: μ (Auxiliary Control Input)
    
#     Objective:
#         min (μ)  ½ μᵀ H μ + fᵀ μ
        
#     Subject to:
    
#     1. Stability Constraint (CLF):
#        LgV(x) μ ≤ -γ V(x) - LfV(x)
       
#     2. Input Constraints (Torque Saturation):
#        τ_min ≤ M(q)J†(u_ref + μ - J̇q̇) + n(q,q̇) ≤ τ_max
#        (Rearranged into linear form A_torq μ ≤ b_torq)
       
#     3. Safety Constraints (CBF - Optional):
#        A_cbf μ ≤ b_cbf
#     """
    
#     # ---------------------------------------------------------
#     # 1. Objective Function
#     #    Minimize control deviation energy: ||μ||²
#     #    Standard Form: min ½ xᵀ P x + qᵀ x
#     # ---------------------------------------------------------
#     # 3 variables for μ (x, y, z acceleration correction)
#     n_vars = 3
    
#     # H = 2I  (Factor of 2 because standard form is 1/2 xPx)
#     H = 2.0 * np.eye(n_vars)
    
#     # f = 0
#     f = np.zeros(n_vars)

#     # ---------------------------------------------------------
#     # 2. Build Constraints (G x ≤ h)
#     # ---------------------------------------------------------
#     A_list = []
#     b_list = []

#     # --- Constraint A: RES-CLF Stability ---
#     # Inequality: LgV ⋅ μ ≤ -γ V - LfV
#     A_list.append(LgV) 
#     b_list.append(np.array([[-gamma * V - LfV]]))

#     # --- Constraint B: Torque Saturation ---
#     # Form: A_torq ⋅ μ ≤ b_torq
#     # (Matrices pre-calculated in main controller to save time here)
#     if torque_A is not None and torque_b is not None:
#         A_list.append(torque_A)
#         b_list.append(torque_b)

#     # --- Constraint C: Control Barrier Function (Safety) ---
#     # Form: A_cbf ⋅ μ ≤ b_cbf
#     if cbf_A is not None and cbf_b is not None:
#         A_list.append(cbf_A)
#         b_list.append(cbf_b)

#     # 3. Stack Matrices for Solver
#     if not A_list:
#         return np.zeros(n_vars), True

#     G_qp = np.vstack(A_list)
#     h_qp = np.vstack(b_list)

#     # 4. Solve using CVXOPT
#     # Solves: min ½ xᵀ P x + qᵀ x  s.t.  G x ≤ h
#     cvxopt.solvers.options['show_progress'] = False
#     P_cvx = cvxopt.matrix(H)
#     q_cvx = cvxopt.matrix(f)
#     G_cvx = cvxopt.matrix(G_qp)
#     h_cvx = cvxopt.matrix(h_qp)

#     try:
#         sol = cvxopt.solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
#         mu = np.array(sol['x']).flatten()
#         return mu, True
        
#     except ValueError:
#         # Infeasible (Constraints conflict, e.g., Safety vs Limits)
#         # Returns zero correction (Fall back to nominal control)
#         return np.zeros(n_vars), False