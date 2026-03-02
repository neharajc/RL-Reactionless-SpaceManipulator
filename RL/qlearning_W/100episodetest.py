import mujoco
import numpy as np
import pickle
from collections import defaultdict

# -------------------------
# Load saved Q-table
# -------------------------
with open("90percentsucess.pkl", "rb") as f:
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

# -------------------------
# Run ONE episode (returns success or failure)
# -------------------------
def run_episode(steps=800, target1=0.0, target2=0.0):

    mujoco.mj_resetData(model, data)

    # Random initial position
    data.qpos[j1_id] = np.radians(np.random.uniform(0, 10))
    data.qpos[j2_id] = np.radians(np.random.uniform(0, 10))
    mujoco.mj_forward(model, data)

    for _ in range(steps):

        j1 = np.degrees(data.qpos[j1_id])
        j2 = np.degrees(data.qpos[j2_id])

        # Check success
        if abs(j1 - target1) + abs(j2 - target2) < 0.5:
            return True   # ✅ success

        s_key = (
            angle_to_index(j1),
            angle_to_index(j2)
        )

        a_idx = choose_best_action(s_key)
        tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

        data.ctrl[1] = tau1
        data.ctrl[2] = tau2
        data.ctrl[3] = tau3
        data.ctrl[4] = tau4

        mujoco.mj_step(model, data)

    return False  # ❌ failed to reach target

# -------------------------
# Evaluate over N episodes
# -------------------------
def evaluate_policy(num_episodes=100):

    success_count = 0

    for ep in range(num_episodes):
        success = run_episode()
        success_count += int(success)
        print(f"Episode {ep+1:03d} | {'SUCCESS' if success else 'FAIL'}")

    success_rate = 100.0 * success_count / num_episodes

    print("\n==============================")
    print(f"Total Episodes : {num_episodes}")
    print(f"Successful     : {success_count}")
    print(f"Success Rate   : {success_rate:.2f}%")
    print("==============================\n")

# -------------------------
if __name__ == "__main__":
    evaluate_policy(num_episodes=100)
