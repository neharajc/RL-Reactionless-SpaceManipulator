import numpy as np
import pickle

def generate_initial_state():
    """
    Generates a random initial joint configuration (degrees)

    joint1, joint2 ∈ [0, 10] deg
    joint3, joint4 = 0 deg
    """
    q_init = np.array([
        np.random.uniform(0.0, 10.0),  # joint1
        np.random.uniform(0.0, 10.0),  # joint2
        0.0,                           # joint3
        0.0                            # joint4
    ])

    return q_init


# -------------------------
# Generate and save
# -------------------------
q_init_deg = generate_initial_state()

with open("initial_state.pkl", "wb") as f:
    pickle.dump(q_init_deg, f)

print("Initial state saved to initial_state.pkl")
print("Initial joint angles (deg):", q_init_deg)
