import numpy as np
import pickle

def generate_initial_state(seed=None):
    """
    Generates a shared initial joint configuration (degrees)

    joint1, joint2 ∈ [0, 10] deg
    joint3, joint4 = 0 deg
    """
    if seed is not None:
        np.random.seed(seed)

    q_init = np.array([
        np.random.uniform(0.0, 10.0),  # joint1
        np.random.uniform(0.0, 10.0),  # joint2
        0.0,                           # joint3
        0.0                            # joint4
    ])

    return q_init


# -------------------------
# Generate & save
# -------------------------
q_init_deg = generate_initial_state(seed=7)

init_state = {
    "q_init_deg": q_init_deg,
    "q_init_rad": np.radians(q_init_deg)
}

with open("initial_state.pkl", "wb") as f:
    pickle.dump(init_state, f)

print("Initial state saved to initial_state.pkl")
print("Initial joint angles (deg):", q_init_deg)
