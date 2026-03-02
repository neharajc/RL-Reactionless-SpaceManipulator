import numpy as np
import pickle

init_state = {
    "q_start": np.array([
        np.radians(np.random.uniform(0, 10)),  # joint1
        np.radians(np.random.uniform(0, 10)),  # joint2
        0.0,                                   # joint3
        0.0                                    # joint4
    ]),
    "q_target": np.zeros(4)
}

with open("init_state.pkl", "wb") as f:
    pickle.dump(init_state, f)

print("Initial condition saved (degrees):")
print(np.degrees(init_state["q_start"]))
print("Target (degrees):", np.degrees(init_state["q_target"]))
