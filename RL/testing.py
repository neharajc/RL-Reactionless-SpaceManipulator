# test_q_table.py

import mujoco
import mujoco.viewer
import numpy as np
import pickle
import time

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

# -------------------------
# Load trained Q-table
# -------------------------
with open("q_table_81.pkl", "rb") as f:
    Q_table = pickle.load(f)

# -------------------------
# Discretization function
# -------------------------
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 10.0, 0.1
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))

def angle_to_index(angle):
    a = max(min(angle, ANGLE_MAX), ANGLE_MIN)
    idx = int((a - ANGLE_MIN) / STEP_DEG)
    return max(0, min(BINS-1, idx))

# -------------------------
# Testing loop using Q-table
# -------------------------
def test_q_table(steps=200):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Random initial positions for joints 1 & 2
        data.qpos[j1_id] = np.radians(np.random.uniform(0, 10))
        data.qpos[j2_id] = np.radians(np.random.uniform(0, 10))
        mujoco.mj_forward(model, data)

        target1 = 0.0
        target2 = 0.0
        total_reward = 0.0

        for step in range(steps):
            j1 = np.degrees(data.qpos[j1_id])
            j2 = np.degrees(data.qpos[j2_id])

            # State key
            s_key = (angle_to_index(j1), angle_to_index(j2))

            # Choose best action from Q-table
            qvals = [Q_table.get((s_key, a_idx), 0.0) for a_idx in range(81)]
            a_idx = int(np.argmax(qvals))

            # Decode 81-action index to 4 joint torques
            base = 3
            tau1 = [-1.0,0.0,1.0][(a_idx // (base**3)) % 3]
            tau2 = [-1.0,0.0,1.0][(a_idx // (base**2)) % 3]
            tau3 = [-1.0,0.0,1.0][(a_idx // (base**1)) % 3]
            tau4 = [-1.0,0.0,1.0][(a_idx // (base**0)) % 3]

            # Apply torques (reactionless)
            data.qfrc_applied[j1_id] = tau1
            data.qfrc_applied[j2_id] = tau2
            data.qfrc_applied[j3_id] = tau3
            data.qfrc_applied[j4_id] = tau4

            # Step simulation
            mujoco.mj_step(model, data)

            # Compute reward
            err1 = abs(j1 - target1)
            err2 = abs(j2 - target2)
            reward = - (err1 + err2)
            total_reward += reward

            # Render
            viewer.sync()
            time.sleep(0.02)

            # Print status
            print(f"Step {step}: J1={j1:.2f}°, J2={j2:.2f}°, Reward={reward:.2f}")

            # Stop if close enough
            if err1 + err2 < 0.1:
                print(f"✅ Target reached at step {step}")
                break

        print(f"Total reward for this episode: {total_reward:.2f}")

# -------------------------
if __name__ == "__main__":
    test_q_table()

