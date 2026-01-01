import numpy as np
from scipy.optimize import minimize

def solve_qp(LfV, LgV, V_val, gamma, e, de):
    """
    Module 3: Optimization.
    Solves Eq (16): min ||mu||^2 s.t. LfV + LgV*mu <= -gamma*V
    """
    
    # Inequality Constraint: LfV + LgV*mu <= -gamma * V
    # Rearranged for solver (cons >= 0): 
    # -LgV*mu - LfV - gamma*V >= 0
    # OR: bound - LgV*mu >= 0, where bound = -gamma*V - LfV
    
    bound = -gamma * V_val - LfV
    
    def constraint_func(mu):
        return bound - (LgV @ mu)
    
    # Objective: Minimize norm of mu
    def cost_func(mu):
        return 0.5 * np.sum(mu**2)
    
    # Initial guess
    mu0 = np.zeros(3)
    
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Use SLSQP solver
    res = minimize(cost_func, mu0, constraints=cons, method='SLSQP')
    
    if res.success:
        return res.x
    else:
        # Robust Fallback if QP fails
        return -10.0 * e - 5.0 * de