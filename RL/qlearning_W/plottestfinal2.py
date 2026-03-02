import mujoco
import mujoco.viewer
import numpy as np
import pickle
from collections import defaultdict
import time
import matplotlib.pyplot as plt

# -------------------------
# Load saved Q-table
# -------------------------
with open("final.pkl", "rb") as f:
    Q_loaded = pickle.load(f)

Q = defaultdict(float, Q_loaded)

# -------------------------
# Load initial condition
# -------------------------
with open("init_state.pkl", "rb") as f:
    INIT = pickle.load(f)

q_start = INIT["q_start"]   # already in radians
q_target = INIT["q_target"] # target (radians)

# -------------------------
# Action definitions
# -------------------------
TORQUE_LEVELS = [-2.5, 0.0, 2.5]
ACTIONS = [(t1, t2, t3, t4)
           for t1 in TORQUE_LEVELS
           for t2 in TORQUE_LEVELS
           for t3 in TORQUE_LEVELS
           for t4 in TORQUE_LEVELS]

# -------------------------
# Discretisation
# -------------------------
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 10.0, 0.5
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))

def angle_to_index(angle):
    a = max(min(angle, ANGLE_MAX), ANGLE_MIN)
    idx = int((a - ANGLE_MIN) / STEP_DEG)
    return max(0, min(BINS - 1, idx))

# -------------------------
# Choose best action
# -------------------------
def choose_best_action(state_key):
    qvals = [Q[(state_key, idx)] for idx in range(len(ACTIONS))]
    return int(np.argmax(qvals))

# -------------------------
# Load MuJoCo model
# -------------------------
model = mujoco.MjModel.from_xml_path("nullspace.xml")
data = mujoco.MjData(model)

# Joint IDs
j1_id = model.joint("joint1").id
j2_id = model.joint("joint2").id
j3_id = model.joint("joint3").id
j4_id = model.joint("joint4").id

# Base joint indices
base_joint = model.joint("base_joint")
base_qpos_adr = base_joint.qposadr[0]
base_dof_adr  = base_joint.dofadr[0]

# -------------------------
# Test function
# -------------------------
def test_agent(steps=800):

    mujoco.mj_resetData(model, data)

    # -------------------------
    # Use initial state from init_state.pkl
    # -------------------------
    data.qpos[j1_id] = q_start[0]
    data.qpos[j2_id] = q_start[1]
    data.qpos[j3_id] = q_start[2]
    data.qpos[j4_id] = q_start[3]

    mujoco.mj_forward(model, data)

    print("Initial joints (deg):", np.degrees(data.qpos[[j1_id, j2_id, j3_id, j4_id]]))

    # -------------------------
    # Logging
    # -------------------------
    time_log = []
    j1_err_log = []
    j2_err_log = []
    tau1_log = []
    tau2_log = []
    tau3_log = []
    tau4_log = []
    base_vel_log = []
    base_pos_log = []

    print("\n========== TEST START ==========")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(steps):

            # Joint angles (deg)
            j1 = np.degrees(data.qpos[j1_id])
            j2 = np.degrees(data.qpos[j2_id])
            j3 = np.degrees(data.qpos[j3_id])
            j4 = np.degrees(data.qpos[j4_id])
            base_pos = data.qpos[base_qpos_adr]

            print(
                f"Step {step:03d} | "
                f"j1={j1:.3f}°, j2={j2:.3f}°, "
                f"j3={j3:.3f}°, j4={j4:.3f}° | "
                f"base_qpos={base_pos:.6f} rad"
            )

            # Errors
            j1_err = abs(j1 - np.degrees(q_target[0]))
            j2_err = abs(j2 - np.degrees(q_target[1]))

            # Log
            time_log.append(step)
            j1_err_log.append(j1_err)
            j2_err_log.append(j2_err)

            # Stop condition
            if j1_err + j2_err < 0.5:
                print("🎉 TARGET REACHED!")
                break

            # State
            s_key = (
                angle_to_index(j1),
                angle_to_index(j2)
            )

            # Action
            a_idx = choose_best_action(s_key)
            tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

            # Log action
            tau1_log.append(tau1)
            tau2_log.append(tau2)
            tau3_log.append(tau3)
            tau4_log.append(tau4)

            # Apply action
            data.ctrl[1] = tau1
            data.ctrl[2] = tau2
            data.ctrl[3] = tau3
            data.ctrl[4] = tau4

            # Step simulation
            mujoco.mj_step(model, data)

            # Base velocity & position
            base_vel_log.append(data.qvel[base_dof_adr])
            base_pos_log.append(data.qpos[base_qpos_adr])

            viewer.sync()
            time.sleep(0.01)

    print("========== TEST END ==========\n")

    # -------------------------
    # Plot joint error
    # -------------------------
    plt.figure()
    plt.plot(time_log, j1_err_log)
    plt.plot(time_log, j2_err_log)
    plt.xlabel("Step")
    plt.ylabel("Joint Error (deg)")
    plt.title("Joint Error vs Time")
    plt.legend(["j1 error", "j2 error"])
    plt.grid()
    plt.show()

    # -------------------------
    # Plot actions
    # -------------------------
    plt.figure()
    plt.plot(time_log[:len(tau1_log)], tau1_log)
    plt.plot(time_log[:len(tau2_log)], tau2_log)
    plt.plot(time_log[:len(tau3_log)], tau3_log)
    plt.plot(time_log[:len(tau4_log)], tau4_log)
    plt.xlabel("Step")
    plt.ylabel("Torque")
    plt.title("Action (Torque) vs Time")
    plt.legend(["tau1", "tau2", "tau3", "tau4"])
    plt.grid()
    plt.show()

    # -------------------------
    # Plot base velocity
    # -------------------------
    plt.figure()
    plt.plot(time_log[:len(base_vel_log)], base_vel_log)
    plt.xlabel("Step")
    plt.ylabel("Base Velocity")
    plt.title("Base Velocity vs Time")
    plt.grid()
    plt.show()

    # -------------------------
    # Plot base position
    # -------------------------
    plt.figure()
    plt.plot(time_log[:len(base_pos_log)], base_pos_log)
    plt.xlabel("Step")
    plt.ylabel("Base Position (qpos)")
    plt.title("Base Position vs Time")
    plt.grid()
    plt.show()


# -------------------------
if __name__ == "__main__":
    test_agent()

