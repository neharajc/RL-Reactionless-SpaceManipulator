import mujoco
import mujoco.viewer
import numpy as np
import random, math, time
from collections import defaultdict
import pickle

# -------------------------
# Discretisation
# -------------------------
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 10.0, 0.1   # restrict to 0–10 degrees
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))  # 100 bins

def angle_to_index(angle):
    a = angle % 360.0
    idx = int(math.floor((a - ANGLE_MIN) / STEP_DEG))
    return max(0, min(BINS-1, idx))

def discretize_state(qpos_old, qpos_new, threshold=0.05):
    """Ignore tiny movements smaller than threshold"""
    if abs(qpos_new - qpos_old) < threshold:
        return qpos_old
    else:
        return qpos_new

def abs_angle_diff_deg(a, b):
    """Compute absolute shortest distance between two angles"""
    diff = abs((a - b + 180) % 360 - 180)
    return diff

# -------------------------
# Actions for 4 joints
# -------------------------
TORQUE_LEVELS = [-1.0, 0.0, 1.0]
ACTIONS = [(t1, t2, t3, t4) for t1 in TORQUE_LEVELS
                               for t2 in TORQUE_LEVELS
                               for t3 in TORQUE_LEVELS
                               for t4 in TORQUE_LEVELS]  # 81 actions

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
# Training loop with reactionless and base-penalized reward
# -------------------------
def train_demo(episodes=20, steps=200):
    agent = QLearnerSparse(ACTIONS, alpha=0.2, gamma=0.99, eps=0.3)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(episodes):
            mujoco.mj_resetData(model, data)

            total_reward = 0.0

            # -------------------------
            # Start j1, j2 from random position (0–10°)
            # -------------------------
            init_j1 = np.radians(random.uniform(0, 10))
            init_j2 = np.radians(random.uniform(0, 10))
            data.qpos[j1_id] = init_j1
            data.qpos[j2_id] = init_j2
            mujoco.mj_forward(model, data)  # apply initial state

            # Fixed home target (always 0°)
            target1 = 0.0
            target2 = 0.0

            # Previous joint positions for thresholding
            j1_prev = np.degrees(data.qpos[j1_id])
            j2_prev = np.degrees(data.qpos[j2_id])
            j3_prev = np.degrees(data.qpos[j3_id])
            j4_prev = np.degrees(data.qpos[j4_id])

            # Previous error for progress reward
            prev_err = abs_angle_diff_deg(j1_prev, target1) + abs_angle_diff_deg(j2_prev, target2)

            for step in range(steps):
                # Current joint angles
                j1 = np.degrees(data.qpos[j1_id])
                j2 = np.degrees(data.qpos[j2_id])
                j3 = np.degrees(data.qpos[j3_id])
                j4 = np.degrees(data.qpos[j4_id])

                # Threshold discretization
                j1 = discretize_state(j1_prev, j1)
                j2 = discretize_state(j2_prev, j2)
                j3 = discretize_state(j3_prev, j3)
                j4 = discretize_state(j4_prev, j4)
                j1_prev, j2_prev, j3_prev, j4_prev = j1, j2, j3, j4

                # Q-table key (discretize only joints 1 & 2)
                s_key = (angle_to_index(j1), angle_to_index(j2))

                # Choose action
                a_idx = agent.choose_action(s_key)
                tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

                # Apply torques
                data.qfrc_applied[j1_id] = tau1
                data.qfrc_applied[j2_id] = tau2
                data.qfrc_applied[j3_id] = tau3
                data.qfrc_applied[j4_id] = tau4

                # Step simulation
                mujoco.mj_step(model, data)

                # -------------------------
                # Reward function: target-reaching + progress + base penalty
                # -------------------------
                j1_next = np.degrees(data.qpos[j1_id])
                j2_next = np.degrees(data.qpos[j2_id])
                err1 = abs_angle_diff_deg(j1_next, target1)
                err2 = abs_angle_diff_deg(j2_next, target2)
                err = err1 + err2

                # Progress reward
                reward_progress = prev_err - err

                # Small step penalty
                step_penalty = -0.01

                # Base reaction penalty (linear + angular velocity of base body)
                base_lin_vel = np.linalg.norm(data.qvel[0:3])   # base translation
                base_ang_vel = np.linalg.norm(data.qvel[3:6])   # base rotation
                base_reaction = base_lin_vel + base_ang_vel
                base_penalty = -0.1 * base_reaction

                # Total reward
                reward = reward_progress + step_penalty + base_penalty

                # Terminal check
                tolerance = 0.5
                done = False
                if err < tolerance:
                    reward += 100.0
                    done = True

                prev_err = err  # update prev_err

                # Next state
                s_next_key = (angle_to_index(j1_next), angle_to_index(j2_next))

                # Q-learning update
                agent.update(s_key, a_idx, reward, s_next_key, done)
                total_reward += reward
                j1_angle = np.degrees(data.qpos[j1_id])
                j2_angle = np.degrees(data.qpos[j2_id])
                j3_angle = np.degrees(data.qpos[j3_id])
                j4_angle = np.degrees(data.qpos[j4_id])

                print(f"J1={j1_angle:.2f}°, J2={j2_angle:.2f}°, J3={j3_angle:.2f}°, J4={j4_angle:.2f}°")

                # Render
                viewer.sync()
                time.sleep(0.02)

            print(f"Episode {ep:03d}  total_reward={total_reward:.2f}  eps={agent.eps:.3f}")

            # Decay epsilon
            agent.eps = max(0.01, agent.eps * 0.99)

    # Save learned Q
    with open("q_sparse_j1j2.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)
    print("Training complete, Q saved.")

# -------------------------
# Run + Keep viewer open
# -------------------------
if __name__ == "__main__":
    train_demo(episodes=20, steps=200)

    print("Simulation finished. Viewer running...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)
