import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# -------------------------
# Load model
# -------------------------
model = mujoco.MjModel.from_xml_path("nullspace.xml")
data = mujoco.MjData(model)

# Joint indices
j1_id = model.joint("joint1").id
j2_id = model.joint("joint2").id
j3_id = model.joint("joint3").id
j4_id = model.joint("joint4").id
base_dof = model.joint("base_joint").dofadr[0]

# -------------------------
# Discretization
# -------------------------
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 10.0, 0.5
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))

def angle_to_index(angle):
    a = max(min(angle, ANGLE_MAX), ANGLE_MIN)
    idx = int(np.floor((a - ANGLE_MIN) / STEP_DEG))
    return max(0, min(BINS-1, idx))

# -------------------------
# Actions
# -------------------------
TORQUE_LEVELS = [-5.0, 0.0, 5.0]
ACTIONS = [(t1,t2,t3,t4) for t1 in TORQUE_LEVELS
                        for t2 in TORQUE_LEVELS
                        for t3 in TORQUE_LEVELS
                        for t4 in TORQUE_LEVELS]

# -------------------------
# Load Q-table
# -------------------------
with open("q_table_81.pkl", "rb") as f:
    Q = pickle.load(f)

# -------------------------
# Testing / rollout
# -------------------------
episodes = 5
steps = 750

# Logging arrays
all_j1, all_j2, all_base_vel, all_rewards = [], [], [], []

with mujoco.viewer.launch_passive(model, data) as viewer:
    for ep in range(episodes):
        mujoco.mj_resetData(model, data)
        # Random start for joints 1 & 2
        data.qpos[j1_id] = np.radians(np.random.uniform(0,10))
        data.qpos[j2_id] = np.radians(np.random.uniform(0,10))
        mujoco.mj_forward(model, data)

        ep_rewards = []

        for step in range(steps):
            j1 = np.degrees(data.qpos[j1_id])
            j2 = np.degrees(data.qpos[j2_id])
            base_vel = data.qvel[base_dof]

            s_key = (angle_to_index(j1), angle_to_index(j2))
            qvals = [Q.get((s_key, idx), 0.0) for idx in range(len(ACTIONS))]
            best_idx = int(np.argmax(qvals))
            tau1, tau2, tau3, tau4 = ACTIONS[best_idx]

            # Apply torques
            data.ctrl[0] = tau1
            data.ctrl[1] = tau2
            data.ctrl[2] = tau3
            data.ctrl[3] = tau4

            # Step simulation
            mujoco.mj_step(model, data)

            # Next angles
            j1_next = np.degrees(data.qpos[j1_id])
            j2_next = np.degrees(data.qpos[j2_id])
            curr_err = abs(j1_next) + abs(j2_next)

            # Reward calculation (optional)
            prev_err = abs(j1) + abs(j2)
            step_penalty = -0.01
            base_penalty = -0.1 * abs(base_vel)
            reward = (prev_err - curr_err) + step_penalty + base_penalty
            if curr_err < 0.5:
                reward += 100.0
                done = True
            else:
                done = False

            ep_rewards.append(reward)

            # Log
            all_j1.append(j1_next)
            all_j2.append(j2_next)
            all_base_vel.append(base_vel)
            all_rewards.append(reward)

            # Render
            viewer.sync()
            time.sleep(0.01)

            if done:
                break

        print(f"Episode {ep} finished | Total Reward={sum(ep_rewards):.2f}")

# -------------------------
# Visualizations
# -------------------------

# 1. Joint trajectories
plt.figure(figsize=(10,4))
plt.plot(all_j1, label='Joint1 (deg)')
plt.plot(all_j2, label='Joint2 (deg)')
plt.xlabel('Step')
plt.ylabel('Angle (deg)')
plt.title('Joint trajectories')
plt.legend()
plt.show()

# 2. Base angular velocity
plt.figure(figsize=(10,4))
plt.plot(all_base_vel)
plt.xlabel('Step')
plt.ylabel('Base angular velocity')
plt.title('Base velocity during rollout')
plt.show()

# 3. Reward over time
plt.figure(figsize=(10,4))
plt.plot(all_rewards)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward per step')
plt.show()

# 4. Max Q-value heatmap
heatmap = np.zeros((BINS, BINS))
for bin_j1 in range(BINS):
    for bin_j2 in range(BINS):
        state_key = (bin_j1, bin_j2)
        qvals = [Q.get((state_key, a_idx), 0.0) for a_idx in range(len(ACTIONS))]
        heatmap[bin_j1, bin_j2] = np.max(qvals)

plt.figure(figsize=(6,5))
plt.imshow(heatmap, origin='lower', aspect='auto')
plt.colorbar(label='Max Q-value')
plt.xlabel('Joint2 bin')
plt.ylabel('Joint1 bin')
plt.title('Max Q-value heatmap')
plt.show()
