import mujoco
import mujoco.viewer
import numpy as np
import random, time
from collections import defaultdict
import pickle

# -------------------------
# Discretisation
# -------------------------
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 10.0, 0.5
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))

def angle_to_index(angle, prev_angle=None, threshold=0.5):
    a = max(min(angle, ANGLE_MAX), ANGLE_MIN)
    idx = int((a - ANGLE_MIN) / STEP_DEG)
    idx = max(0, min(BINS - 1, idx))

    if prev_angle is not None:
        prev_idx = int((prev_angle - ANGLE_MIN) / STEP_DEG)
        if abs(idx - prev_idx) * STEP_DEG < threshold:
            idx = prev_idx
    return idx

# -------------------------
# Actions (4 joints, base passive)
# -------------------------
TORQUE_LEVELS = [-2.5, 0.0, 2.5]
ACTIONS = [(t1, t2, t3, t4)
           for t1 in TORQUE_LEVELS
           for t2 in TORQUE_LEVELS
           for t3 in TORQUE_LEVELS
           for t4 in TORQUE_LEVELS]

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
        qvals = [self.get_q(s_key, i) for i in range(len(self.actions))]
        return int(np.argmax(qvals))

    def update(self, s_key, a_idx, reward, s_next_key, done):
        q_sa = self.get_q(s_key, a_idx)
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(
                self.get_q(s_next_key, i) for i in range(len(self.actions))
            )
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

# Base DOF index (passive)
base_dof = model.joint("base_joint").dofadr[0]

# -------------------------
# Training loop
# -------------------------
def train_demo(episodes=30000, steps=750):
    agent = QLearnerSparse(ACTIONS)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(episodes):
            mujoco.mj_resetData(model, data)

            # Random start
            data.qpos[j1_id] = np.radians(random.uniform(0, 10))
            data.qpos[j2_id] = np.radians(random.uniform(0, 10))
            mujoco.mj_forward(model, data)

            j1_prev = np.degrees(data.qpos[j1_id])
            j2_prev = np.degrees(data.qpos[j2_id])

            total_reward = 0.0
            target1, target2 = 0.0, 0.0

            for step in range(steps):
                j1 = np.degrees(data.qpos[j1_id])
                j2 = np.degrees(data.qpos[j2_id])

                prev_err = abs(j1 - target1) + abs(j2 - target2)

                s_key = (
                    angle_to_index(j1, prev_angle=j1_prev),
                    angle_to_index(j2, prev_angle=j2_prev)
                )

                a_idx = agent.choose_action(s_key)
                tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

                # Apply torques (base untouched)
                data.ctrl[1] = tau1
                data.ctrl[2] = tau2
                data.ctrl[3] = tau3
                data.ctrl[4] = tau4

                mujoco.mj_step(model, data)

                j1_next = np.degrees(data.qpos[j1_id])
                j2_next = np.degrees(data.qpos[j2_id])

                base_vel = data.qvel[base_dof]

                curr_err = abs(j1_next - target1) + abs(j2_next - target2)
                error_term = prev_err - curr_err
                step_penalty = -0.01

                # --------------------------------
                # Soft base penalty (goal-aware)
                # --------------------------------
                if curr_err < 3.0:
                    base_tol = 0.05
                    base_penalty = -0.3 * max(0.0, abs(base_vel) - base_tol) ** 2
                else:
                    base_penalty = 0.0

                reward = error_term + step_penalty + base_penalty

                done = False
                if curr_err < 0.75:
                    reward += 100.0
                    done = True

                s_next_key = (
                    angle_to_index(j1_next, prev_angle=j1),
                    angle_to_index(j2_next, prev_angle=j2)
                )

                agent.update(s_key, a_idx, reward, s_next_key, done)
                total_reward += reward

                j1_prev, j2_prev = j1_next, j2_next

                viewer.sync()
                time.sleep(0.01)

                if done:
                    break

            # Slower ε decay
            agent.eps = max(0.02, agent.eps * 0.995)

            print(f"Episode {ep:05d} | Reward={total_reward:.2f} | eps={agent.eps:.3f}")

    with open("q_table_81_passive_base.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)

    print("✅ Training complete. Q-table saved.")

# -------------------------
if __name__ == "__main__":
    train_demo()
