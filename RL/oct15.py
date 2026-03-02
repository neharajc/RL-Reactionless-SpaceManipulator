import mujoco
import mujoco.viewer
import numpy as np
import random, math, time
from collections import defaultdict
import pickle

# -------------------------
# Discretisation (0°..10° with 0.1° bins)
# -------------------------
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 10.0, 0.1
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))  # 100

def angle_to_index(angle):
    """Convert angle (deg) to discrete index 0..BINS-1"""
    a = max(ANGLE_MIN, min(ANGLE_MAX, abs(angle)))  # only positive range 0–10
    idx = int(math.floor((a - ANGLE_MIN) / STEP_DEG))
    return max(0, min(BINS - 1, idx))

def abs_angle_diff_deg(a, b):
    """Shortest absolute difference between two angles in degrees."""
    return abs(((a - b + 180.0) % 360.0) - 180.0)

# -------------------------
# Action space: all 4 joints (3×3×3×3 = 81)
# -------------------------
TORQUE_LEVELS = [-1.0, 0.0, 1.0]
ACTIONS = [(t1, t2, t3, t4)
           for t1 in TORQUE_LEVELS
           for t2 in TORQUE_LEVELS
           for t3 in TORQUE_LEVELS
           for t4 in TORQUE_LEVELS]  # 81 actions

# -------------------------
# Q-Learner (sparse dictionary)
# -------------------------
class QLearnerSparse:
    def __init__(self, actions, alpha=0.2, gamma=0.99, eps=0.3):
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
        qvals = [self.get_q(state_key, i) for i in range(len(self.actions))]
        return int(np.argmax(qvals))

    def update(self, s_key, action_idx, reward, s_next_key, done):
        q_sa = self.get_q(s_key, action_idx)
        if done:
            target = reward
        else:
            q_next = max(self.get_q(s_next_key, i) for i in range(len(self.actions)))
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
def train_q_learning(episodes=100, steps=200):
    agent = QLearnerSparse(ACTIONS, alpha=0.2, gamma=0.99, eps=0.3)
    rewards_history = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(episodes):
            mujoco.mj_resetData(model, data)

            # Randomize start for joint1 & joint2 (0–10°)
            data.qpos[j1_id] = np.radians(random.uniform(0, 10))
            data.qpos[j2_id] = np.radians(random.uniform(0, 10))
            mujoco.mj_forward(model, data)

            total_reward = 0.0

            prev_err = abs_angle_diff_deg(np.degrees(data.qpos[j1_id]), 0.0) \
                     + abs_angle_diff_deg(np.degrees(data.qpos[j2_id]), 0.0)

            for step in range(steps):
                # Current angles
                j1 = np.degrees(data.qpos[j1_id])
                j2 = np.degrees(data.qpos[j2_id])

                # Discretize state (only joint1 & joint2)
                s_key = (angle_to_index(j1), angle_to_index(j2))

                # Choose action (4 torques)
                a_idx = agent.choose_action(s_key)
                tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

                # Apply torques
                data.qfrc_applied[j1_id] = tau1
                data.qfrc_applied[j2_id] = tau2
                data.qfrc_applied[j3_id] = tau3
                data.qfrc_applied[j4_id] = tau4

                mujoco.mj_step(model, data)

                # Next state
                j1_next = np.degrees(data.qpos[j1_id])
                j2_next = np.degrees(data.qpos[j2_id])
                err1 = abs_angle_diff_deg(j1_next, 0.0)
                err2 = abs_angle_diff_deg(j2_next, 0.0)
                err = err1 + err2

                # Reward: progress + stability + step penalty
                reward_progress = (prev_err - err) / 10.0
                step_penalty = -0.01

                base_lin_vel = np.linalg.norm(data.qvel[0:3]) if data.qvel.shape[0] >= 3 else 0
                base_ang_vel = np.linalg.norm(data.qvel[3:6]) if data.qvel.shape[0] >= 6 else 0
                base_penalty = -0.1 * (base_lin_vel + base_ang_vel)

                reward = reward_progress + step_penalty + base_penalty

                # Termination condition
                done = err < 0.5
                if done:
                    reward += 100.0

                # Q update
                s_next_key = (angle_to_index(j1_next), angle_to_index(j2_next))
                agent.update(s_key, a_idx, reward, s_next_key, done)
                total_reward += reward
                prev_err = err

                # Render simulation
                viewer.sync()
                time.sleep(0.02)

                if done:
                    break

            rewards_history.append(total_reward)
            print(f"Episode {ep:03d} | Total Reward={total_reward:.2f} | eps={agent.eps:.3f}")

            # Decay epsilon
            agent.eps = max(0.01, agent.eps * 0.99)

    # Save learned Q-table
    with open("q_table_81.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)

    print("✅ Training Complete. Q-table saved as q_table_81.pkl")

    return rewards_history, agent

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    train_q_learning(episodes=50, steps=200)

