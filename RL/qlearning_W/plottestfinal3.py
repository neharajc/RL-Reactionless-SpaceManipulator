import mujoco
import mujoco.viewer
import numpy as np
import pickle
from collections import defaultdict
import time
import matplotlib.pyplot as plt

# =========================================================
# Load saved Q-table
# =========================================================
with open("final.pkl", "rb") as f:
    Q_loaded = pickle.load(f)

Q = defaultdict(float, Q_loaded)

# =========================================================
# Load shared initial condition (same as trajectory)
# =========================================================
with open("initial_state.pkl", "rb") as f:
    q_start_deg = pickle.load(f)     # degrees, shape (4,)

q_start = np.radians(q_start_deg)    # radians
q_target = np.zeros(4)               # target always zero (radians)

print("Loaded initial state (deg):", q_start_deg)
print("Target state (deg):", np.degrees(q_target))

# =========================================================
# Action definitions (torque control)
# =========================================================
TORQUE_LEVELS = [-2.5, 0.0, 2.5]
ACTIONS = [
    (t1, t2, t3, t4)
    for t1 in TORQUE_LEVELS
    for t2 in TORQUE_LEVELS
    for t3 in TORQUE_LEVELS
    for t4 in TORQUE_LEVELS
]

# =========================================================
# Discretisation (degrees)
# =========================================================
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 10.0, 0.5
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))

def angle_to_index(angle_deg):
    a = np.clip(angle_deg, ANGLE_MIN, ANGLE_MAX)
    idx = int((a - ANGLE_MIN) / STEP_DEG)
    return np.clip(idx, 0, BINS - 1)

# =========================================================
# Choose greedy action
# =========================================================
def choose_best_action(state_key):
    qvals = [Q[(state_key, i)] for i in range(len(ACTIONS))]
    return int(np.argmax(qvals))

# =========================================================
# Load MuJoCo model
# =========================================================
model = mujoco.MjModel.from_xml_path("nullspace.xml")
data = mujoco.MjData(model)

# Joint IDs
j1_id = model.joint("joint1").id
j2_id = model.joint("joint2").id
j3_id = model.joint("joint3").id
j4_id = model.joint("joint4").id

# Base joint indices (passive)
base_joint = model.joint("base_joint")
base_qpos_adr = base_joint.qposadr[0]
base_dof_adr  = base_joint.dofadr[0]

# =========================================================
# Test rollout
# =========================================================
def test_agent(max_steps=800):

    mujoco.mj_resetData(model, data)

    # -----------------------------------------------------
    # Apply shared initial state
    # -----------------------------------------------------
    data.qpos[j1_id] = q_start[0]
    data.qpos[j2_id] = q_start[1]
    data.qpos[j3_id] = q_start[2]
    data.qpos[j4_id] = q_start[3]

    mujoco.mj_forward(model, data)

    print("\nInitial joints (deg):",
          np.degrees(data.qpos[[j1_id, j2_id, j3_id, j4_id]]))

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------
    time_log = []
    j1_err_log = []
    j2_err_log = []
    base_pos_log = []
    base_vel_log = []

    tau1_log, tau2_log, tau3_log, tau4_log = [], [], [], []

    print("\n========== Q-LEARNING TEST START ==========")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(max_steps):

            # Current state
            j1 = np.degrees(data.qpos[j1_id])
            j2 = np.degrees(data.qpos[j2_id])
            base_pos = data.qpos[base_qpos_adr]

            # Errors
            j1_err = abs(j1)
            j2_err = abs(j2)

            # Log
            time_log.append(step)
            j1_err_log.append(j1_err)
            j2_err_log.append(j2_err)
            base_pos_log.append(base_pos)
            base_vel_log.append(data.qvel[base_dof_adr])

            # Success condition
            if j1_err + j2_err < 0.5:
                print("🎉 TARGET REACHED")
                break

            # Discrete state
            s_key = (
                angle_to_index(j1),
                angle_to_index(j2)
            )

            # Action
            a_idx = choose_best_action(s_key)
            tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

            tau1_log.append(tau1)
            tau2_log.append(tau2)
            tau3_log.append(tau3)
            tau4_log.append(tau4)

            # Apply torques (base unactuated)
            data.ctrl[1] = tau1
            data.ctrl[2] = tau2
            data.ctrl[3] = tau3
            data.ctrl[4] = tau4

            mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(0.01)

    print("========== Q-LEARNING TEST END ==========\n")

    # =====================================================
    # Plots
    # =====================================================
    plt.figure(figsize=(7,5))
    plt.plot(time_log, j1_err_log, linewidth=3, label="Joint 1 Error")
    plt.plot(time_log, j2_err_log, linewidth=3, label="Joint 2 Error")
    plt.xlabel("Step", fontsize=14, fontweight="bold")
    plt.ylabel("Error (deg)", fontsize=14, fontweight="bold")
    plt.title("Joint Tracking Error (Q-Learning)", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(time_log, base_pos_log, linewidth=3)
    plt.xlabel("Step", fontsize=14, fontweight="bold")
    plt.ylabel("Base Position (rad)", fontsize=14, fontweight="bold")
    plt.title("Passive Base Motion (Q-Learning)", fontsize=16, fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =========================================================
if __name__ == "__main__":
    test_agent()


