import mujoco
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

# -------------------------
# Load saved Q-table
# -------------------------
with open("final.pkl", "rb") as f:
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

# Base position index
base_qpos_adr = model.joint("base_joint").qposadr[0]

# -------------------------
# Run ONE episode
# -------------------------
def run_episode(steps=800, target1=0.0, target2=0.0):

    mujoco.mj_resetData(model, data)

    # Random initial joints
    data.qpos[j1_id] = np.radians(np.random.uniform(0, 10))
    data.qpos[j2_id] = np.radians(np.random.uniform(0, 10))
    mujoco.mj_forward(model, data)

    base_pos_init = data.qpos[base_qpos_adr]
    success = False


    for _ in range(steps):

        j1 = np.degrees(data.qpos[j1_id])
        j2 = np.degrees(data.qpos[j2_id])

        if abs(j1 - target1) + abs(j2 - target2) < 0.5:
		        	
            success=True
            break

        s_key = (angle_to_index(j1), angle_to_index(j2))
        a_idx = choose_best_action(s_key)
        tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

        # Base is passive
        data.ctrl[1] = tau1
        data.ctrl[2] = tau2
        data.ctrl[3] = tau3
        data.ctrl[4] = tau4

        mujoco.mj_step(model, data)

    base_pos_final = data.qpos[base_qpos_adr]
    base_error = abs(base_pos_final - base_pos_init)

    return success, base_error

# -------------------------
# Evaluate and plot
# -------------------------
def evaluate_policy(num_episodes=100):

    base_errors = []
    success_count=0

    for ep in range(num_episodes):
        success, err = run_episode()
        base_errors.append(err)
        success_count+=int(success)
        success_rate = 100.0 * success_count / num_episodes
        print(f"Episode {ep+1:03d} | Base Error = {err:.6f} rad {'SUCCESS' if success else 'FAIL'} | ")
        
    	
    print(f"Total Episodes : {num_episodes}")
    print(f"Successful     : {success_count}")
    print(f"Success Rate   : {success_rate:.2f}%")

    plt.figure()
    plt.plot(base_errors, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Base Position Error (rad)")
    plt.title("Base Position Drift per Episode")
    plt.grid()
    plt.show()

# -------------------------
if __name__ == "__main__":
    evaluate_policy(num_episodes=100)
