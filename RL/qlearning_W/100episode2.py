import mujoco
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

# -------------------------
# Load saved Q-table
# -------------------------
with open("newyearq3.pkl", "rb") as f:
    Q_loaded = pickle.load(f)

Q = defaultdict(float, Q_loaded)

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

# Base joint indices
base_qpos_adr = model.joint("base_joint").qposadr[0]

# -------------------------
# Run ONE episode
# -------------------------
def run_episode(steps=800, target1=0.0, target2=0.0):

    mujoco.mj_resetData(model, data)

    # Random initial joint position
    data.qpos[j1_id] = np.radians(np.random.uniform(0, 10))
    data.qpos[j2_id] = np.radians(np.random.uniform(0, 10))
    mujoco.mj_forward(model, data)

    base_pos_init = data.qpos[base_qpos_adr]
    base_pos_log = []

    for _ in range(steps):

        j1 = np.degrees(data.qpos[j1_id])
        j2 = np.degrees(data.qpos[j2_id])

        base_pos = data.qpos[base_qpos_adr]
        base_pos_log.append(base_pos - base_pos_init)

        # Success condition (position only)
        if abs(j1 - target1) + abs(j2 - target2) < 0.5:
            break

        s_key = (
            angle_to_index(j1),
            angle_to_index(j2)
        )

        a_idx = choose_best_action(s_key)
        tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

        # Base is passive (ctrl[0] untouched)
        data.ctrl[1] = tau1
        data.ctrl[2] = tau2
        data.ctrl[3] = tau3
        data.ctrl[4] = tau4

        mujoco.mj_step(model, data)

    base_pos_log = np.array(base_pos_log)

    final_base_error = abs(base_pos_log[-1])
    rms_base_error = np.sqrt(np.mean(base_pos_log ** 2))

    return final_base_error, rms_base_error

# -------------------------
# Evaluate policy
# -------------------------
def evaluate_policy(num_episodes=100):

    final_errors = []
    rms_errors = []

    for ep in range(num_episodes):
        final_err, rms_err = run_episode()
        final_errors.append(final_err)
        rms_errors.append(rms_err)

        print(
            f"Episode {ep+1:03d} | "
            f"Final Base Error = {final_err:.6f} rad | "
            f"RMS Base Error = {rms_err:.6f} rad"
        )

    # -------------------------
    # Plot base error per episode
    # -------------------------
    plt.figure()
    plt.plot(final_errors, label="Final Base Error")
    plt.plot(rms_errors, label="RMS Base Error")
    plt.axhline(0.00825, linestyle="--", label="Stability Threshold (0.00825 rad)")
    plt.xlabel("Episode")
    plt.ylabel("Base Error (rad)")
    plt.title("Base Stability per Episode")
    plt.legend()
    plt.grid()
    plt.show()

# -------------------------
if __name__ == "__main__":
    evaluate_policy(num_episodes=100)
