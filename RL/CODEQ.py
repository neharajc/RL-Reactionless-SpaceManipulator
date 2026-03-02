import mujoco
import mujoco.viewer
import numpy as np
import random, math, time
from collections import defaultdict
import pickle

# -------------------------
# Discretisation
# -------------------------
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 10.0, 0.5   # degrees
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))  # 20 bins

def angle_to_index(angle, prev_angle=None, threshold=0.5):
    
    a = max(min(angle, ANGLE_MAX), ANGLE_MIN)
    idx = int((a - ANGLE_MIN) / STEP_DEG)
    idx = max(0, min(BINS-1, idx))
    
    if prev_angle is not None:
        prev_idx = int((prev_angle - ANGLE_MIN) / STEP_DEG)
        if abs(idx - prev_idx) * STEP_DEG < threshold:
            idx = prev_idx
    return idx

# -------------------------
# Actions: 4 joints, 81 combinations
# -------------------------
TORQUE_LEVELS = [-2.5, 0.0, 2.5]
ACTIONS = [(t1,t2,t3,t4) for t1 in TORQUE_LEVELS
                        for t2 in TORQUE_LEVELS
                        for t3 in TORQUE_LEVELS
                        for t4 in TORQUE_LEVELS]  # 81 actions

# -------------------------
# Q-Learner
# -------------------------
class QLearnerSparse:
    def __init__(self, actions, alpha=0.2, gamma=0.99, eps=0.3):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = defaultdict(float)

    def get_q(self, s_key, a_idx):
        return self.Q[(s_key, a_idx)]

    def choose_action(self, s_key):
        if random.random() < self.eps:
            return random.randrange(len(self.actions))
        qvals = [self.get_q(s_key, idx) for idx in range(len(self.actions))]
        return int(np.argmax(qvals))

    def update(self, s_key, a_idx, reward, s_next_key, done):
        q_sa = self.get_q(s_key, a_idx)
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.get_q(s_next_key, idx) for idx in range(len(self.actions)))
        self.Q[(s_key, a_idx)] = q_sa + self.alpha * (target - q_sa)

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

# Base DOF index
base_dof = model.joint("base_joint").dofadr[0]

# -------------------------
# Training loop
# -------------------------
def train_demo(episodes=40000, steps=750):
    agent = QLearnerSparse(ACTIONS)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(episodes):
            mujoco.mj_resetData(model, data)

            # Random start for joints 1 & 2
            data.qpos[j1_id] = np.radians(random.uniform(0, 10))
            data.qpos[j2_id] = np.radians(random.uniform(0, 10))
            mujoco.mj_forward(model, data)

            # Initialize previous angles for tiny-change suppression
            j1_prev = np.degrees(data.qpos[j1_id])
            j2_prev = np.degrees(data.qpos[j2_id])

            total_reward = 0.0
            target1 = 0.0
            target2 = 0.0

            for step in range(steps):
                # Current joint angles
                j1 = np.degrees(data.qpos[j1_id])
                j2 = np.degrees(data.qpos[j2_id])

                prev_err = abs(j1 - target1) + abs(j2 - target2)

                # -------------------------------
                # State key with tiny-change suppression
                # -------------------------------
                s_key = (angle_to_index(j1, prev_angle=j1_prev),
                         angle_to_index(j2, prev_angle=j2_prev))

                # Choose action
                a_idx = agent.choose_action(s_key)
                tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

                # Apply torques via actuators
                data.ctrl[1] = tau1
                data.ctrl[2] = tau2
                data.ctrl[3] = tau3
                data.ctrl[4] = tau4

                # Step simulation
                mujoco.mj_step(model, data)

                # Next joint angles
                j1_next = np.degrees(data.qpos[j1_id])
                j2_next = np.degrees(data.qpos[j2_id])

                # Base velocity
                base_vel = data.qvel[base_dof]

                # -----------------------------
                # Reward function components
                # -----------------------------
                curr_err = abs(j1_next - target1) + abs(j2_next - target2)
                error_term = prev_err - curr_err       # positive if error reduced
                step_penalty = -0.01
                base_penalty = -2 * abs(base_vel)
                reward = error_term + step_penalty + base_penalty

                # Terminal condition
                done = False
                if curr_err < 0.5:
                    reward += 100.0
                    done = True

                # Next state and Q update (with threshold)
                s_next_key = (angle_to_index(j1_next, prev_angle=j1),
                              angle_to_index(j2_next, prev_angle=j2))

                agent.update(s_key, a_idx, reward, s_next_key, done)
                total_reward += reward

                # Update previous angles
                j1_prev = j1_next
                j2_prev = j2_next

                # Render
                viewer.sync()
                time.sleep(0.01)

                if done:
                    break

            # Decay exploration
            agent.eps = max(0.01, agent.eps * 0.99)

            print(f"Episode {ep:05d} | Total Reward={total_reward:.2f} | eps={agent.eps:.3f}")

    # Save Q-table
    with open("q_table_81improve111111d.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)
    print("✅ Training Complete. Q-table saved as q_tableimproved_81.pkl")

# -------------------------
if __name__ == "__main__":
    train_demo()
