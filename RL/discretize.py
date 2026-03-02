import mujoco
import mujoco.viewer
import numpy as np
import random, math, time
from collections import defaultdict
import pickle

# -------------------------
# Discretisation
# -------------------------
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 360.0, 0.1
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))  # 3600

def angle_to_index(angle):
    a = angle % 360.0
    idx = int(math.floor((a - ANGLE_MIN) / STEP_DEG))
    return max(0, min(BINS-1, idx))

# -------------------------
# Actions
# -------------------------
TORQUE_LEVELS = [-1.0, 0.0, 1.0]   # discrete torque levels
ACTIONS = [(t1, t2) for t1 in TORQUE_LEVELS for t2 in TORQUE_LEVELS]  # 9 actions

# -------------------------
# Q-Learner (sparse dict)
# -------------------------
class QLearnerSparse:
    def __init__(self, actions, alpha=0.1, gamma=0.99, eps=0.2):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = defaultdict(float)

    def get_q(self, state_key, action_idx):
        return self.Q[(state_key, action_idx)]

    def choose_action(self, state_key):
        if random.random() < self.eps:
            return random.randrange(len(self.actions))
        qvals = [self.get_q(state_key, aidx) for aidx in range(len(self.actions))]
        return int(np.argmax(qvals))

    def update(self, s_key, action_idx, reward, s_next_key, done):
        q_sa = self.get_q(s_key, action_idx)
        if done:
            target = reward
        else:
            q_next = max(self.get_q(s_next_key, aidx) for aidx in range(len(self.actions)))
            target = reward + self.gamma * q_next
        self.Q[(s_key, action_idx)] = q_sa + self.alpha * (target - q_sa)

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
# Training loop
# -------------------------
def train_demo(episodes=20, steps=200):
    agent = QLearnerSparse(ACTIONS, alpha=0.2, gamma=0.99, eps=0.3)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(episodes):
            mujoco.mj_resetData(model, data)

            total_reward = 0.0
            for step in range(steps):
                # Current joint angles
                j1 = np.degrees(data.qpos[j1_id])
                j2 = np.degrees(data.qpos[j2_id])
                i1, i2 = angle_to_index(j1), angle_to_index(j2)
                s_key = (i1, i2)

                # Choose action
                a_idx = agent.choose_action(s_key)
                tau1, tau2 = ACTIONS[a_idx]

                # Apply torques on joint1 & joint2
                data.qfrc_applied[j1_id] = tau1
                data.qfrc_applied[j2_id] = tau2

                # Random disturbances on joint3 & joint4
                data.qfrc_applied[j3_id] = np.random.normal(0.0, 0.5)
                data.qfrc_applied[j4_id] = np.random.normal(0.0, 0.5)

                # Step simulation
                mujoco.mj_step(model, data)

                # Reward = keep j1,j2 near 0°
                err1 = abs((j1 % 360) - 0)
                err2 = abs((j2 % 360) - 0)
                reward = - (err1 + err2)

                # Next state
                j1_next = np.degrees(data.qpos[j1_id])
                j2_next = np.degrees(data.qpos[j2_id])
                s_next_key = (angle_to_index(j1_next), angle_to_index(j2_next))

                agent.update(s_key, a_idx, reward, s_next_key, done=False)
                total_reward += reward

                # Render (slow down to ~50 FPS)
                viewer.sync()
                time.sleep(0.02)

            print(f"Episode {ep:03d}  total_reward={total_reward:.2f}  eps={agent.eps:.3f}")

            # decay epsilon
            agent.eps = max(0.01, agent.eps * 0.99)

    # Save learned Q
    with open("q_sparse_j1j2.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)
    print("Training complete, Q saved.")

# -------------------------
# Run + Keep viewer open
# -------------------------
if __name__ == "__main__":
    train_demo()

    # --- Keep window open after training ---
    print("Simulation finished. Viewer running...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)

