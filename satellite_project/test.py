# file: constrained_minimize.py

import numpy as np
from scipy.optimize import minimize

# ---------- Objective function ----------
def objective(x):
    x1, x2, x3, x4 = x
    return (x1 - 2)**2 + (x2 - 4)**2

# ---------- Constraint function ----------
def constraint_eq(x):
    x1, x2, x3, x4 = x
    return 2*x1 + x2 + x3 + x4  # must be 0

# ---------- Main ----------
def main():
    # Initial guess
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    # Define constraint in scipy format
    cons = {'type': 'eq', 'fun': constraint_eq}

    # Solve
    res = minimize(objective, x0, constraints=cons, method='SLSQP')

    if res.success:
        x_opt = res.x
        print("Optimization successful!")
        print("Optimal x1, x2, x3, x4:", np.round(x_opt, 4))
        print("Objective value:", np.round(objective(x_opt), 4))
        print("Constraint value (should be 0):", np.round(constraint_eq(x_opt), 6))
    else:
        print("Optimization failed:", res.message)

if __name__ == "__main__":
    main()
