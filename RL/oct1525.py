import mujoco
import mujoco.viewer
import numpy as np
import random, math, time
from collections import defaultdict
import pickle

# =========================
# Discretization & Config
# =========================
DEG_BIN = 0.5                      # 0.5° bins → 360/0.5 = 720 bins
NUM_BINS = int(round(360.0 / DEG_BIN))
TOL_DEG = 0.5                      # terminal tolerance on total error (err1+err2)
IGNORE_THRESHOLD_DEG = 0.05        # ignore tiny changes for stable Q-updates

# Training horizon (increase for full learning)
EPISODES = 20000
MAX_STEPS = 750

# Q-learning hyperparams
ALPHA = 0.2
GAMMA = 0.995
EPS_START = 1.0
EPS_MIN = 0.01
EPS_DECAY_FRAC = 0.6  # near-min by 60% of training

# Reward shaping
STEP_PENALTY = -0.01
BASE_PENALTY_SCALE = 0.1
TERMINAL_BONUS = 100.0

# Initial ranges for joints 1–2 (degrees)
INIT_RANGE_DEG = (0.0, 10.0)

# Torque action set (denser to align with finer discretization)
TORQUE_LEVELS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
ACTIONS = [(t1,t2,t3,t4) for t1 in TORQUE_LEVELS
                        for t2 in TORQUE_LEVELS
                        for t3 in TORQUE_LEVELS
                        for t4 in TORQUE_LEVELS]  # len = 7^4 = 2401

# =========================
# Helpers
# =========================
def norm_angle_deg(a):
    a = a % 360.0
    return a + 360.0 if a < 0 else a

def angle_to_index(angle_deg):
    a = norm_angle_deg(angle_deg)
    idx = int(a // DEG_BIN)
    if idx >= NUM_BINS:
        idx = NUM_BINS - 1
    return idx

def abs_angle_diff_deg(a, b):
    d = abs(norm_angle_deg(a) - norm_angle_deg(b))
    return min(d, 360.0 - d)

def epsilon_by_episode(ep, episodes=EPISODES, eps_start=EPS_START, eps_min=EPS_MIN, decay_frac=EPS_DECAY_FRAC):
    target_ep = max(1, int(decay_frac * episodes))
    k = math.log(max(eps_start / eps_min, 1.0001)) / target_ep
    eps = eps_start * math.exp(-k * ep)
    return max(eps_min, eps)

# =========================
# Q-learner (sparse)
# =========================
class QLearnerSparse:
    def __init__(self, actions, alpha=ALPHA, gamma=GAMMA, eps=EPS_START):
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
        # greedy with random tie-break
        qvals = [self.get_q(s_key, idx) for idx in range(len(self.actions))]
        maxq = max(qvals)
        candidates = [i for i, v in enumerate(qvals) if v == maxq]
        return random.choice(candidates)

    def update(self, s_key, a_idx, reward, s_next_key, done):
        q_sa = self.get_q(s_key, a_idx)
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.get_q(s_next_key, idx) for idx in range(len(self.actions)))
        self.Q[(s_key, a_idx)] = q_sa + self.alpha * (target - q_sa)

# =========================
# MuJoCo model & indexing
# =========================
model = mujoco.MjModel.from_xml_path("newnull.xml")
data = mujoco.MjData(model)

# Resolve joint ids by name and their qpos/dof addresses
def joint_id(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Joint '{name}' not found in XML")
    return jid

j1_id = joint_id("joint1")
j2_id = joint_id("joint2")
j3_id = joint_id("joint3")
j4_id = joint_id("joint4")

j1_qpos = model.jnt_qposadr[j1_id]
j2_qpos = model.jnt_qposadr[j2_id]
j3_qpos = model.jnt_qposadr[j3_id]
j4_qpos = model.jnt_qposadr[j4_id]

j1_dof = model.jnt_dofadr[j1_id]
j2_dof = model.jnt_dofadr[j2_id]
j3_dof = model.jnt_dofadr[j3_id]
j4_dof = model.jnt_dofadr[j4_id]

# Try to map actuators to these joints; fallback to qfrc_applied by DOF
def actuators_for_joint(jid):
    acts = []
    for a in range(model.nu):
        # For joint transmissions, actuator_trnid[a,0] stores the joint id
        if model.actuator_trnid[a][0] == jid:
            acts.append(a)
    return acts

a1 = actuators_for_joint(j1_id)
a2 = actuators_for_joint(j2_id)
a3 = actuators_for_joint(j3_id)
a4 = actuators_for_joint(j4_id)

use_ctrl = all(len(x) >= 1 for x in [a1, a2, a3, a4])
if use_ctrl:
    a1, a2, a3, a4 = a1[0], a2[0], a3[0], a4[0]

# Find free-joint DOF slices for base velocities (optional)
def base_vel_slices():
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            dof_adr = model.jnt_dofadr[jid]
            return slice(dof_adr, dof_adr+3), slice(dof_adr+3, dof_adr+6)
    return None, None

lin_slice, ang_slice = base_vel_slices()

def get_base_speeds():
    if lin_slice is None:
        return 0.0, 0.0
    lin = float(np.linalg.norm(data.qvel[lin_slice]))
    ang = float(np.linalg.norm(data.qvel[ang_slice]))
    return lin, ang

# =========================
# Training loop
# =========================
def train(episodes=EPISODES, steps=MAX_STEPS, save_path="q_table.pkl", with_viewer=False, viewer_fps_sleep=0.0):
    agent = QLearnerSparse(ACTIONS)
    rng = np.random.default_rng(123)

    def _episode(viewer=None):
        mujoco.mj_resetData(model, data)
        # Randomize joints 1 & 2 within [0,10] deg; others unchanged
        data.qpos[j1_qpos] = np.radians(rng.uniform(*INIT_RANGE_DEG))
        data.qpos[j2_qpos] = np.radians(rng.uniform(*INIT_RANGE_DEG))
        mujoco.mj_forward(model, data)

        target1 = 0.0
        target2 = 0.0

        # Initial angles
        j1 = np.degrees(data.qpos[j1_qpos])
        j2 = np.degrees(data.qpos[j2_qpos])
        prev_err = abs_angle_diff_deg(j1, target1) + abs_angle_diff_deg(j2, target2)

        total_reward = 0.0
        for t in range(steps):
            # ε schedule
            agent.eps = epsilon_by_episode(ep, EPISODES, EPS_START, EPS_MIN, EPS_DECAY_FRAC)

            # State key (discretize j1 & j2 over 360°)
            s_key = (angle_to_index(j1), angle_to_index(j2))

            # Choose action
            a_idx = agent.choose_action(s_key)
            tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

            # Apply torques
            if use_ctrl:
                data.ctrl[a1] = tau1
                data.ctrl[a2] = tau2
                data.ctrl[a3] = tau3
                data.ctrl[a4] = tau4
            else:
                data.qfrc_applied[j1_dof] = tau1
                data.qfrc_applied[j2_dof] = tau2
                data.qfrc_applied[j3_dof] = tau3
                data.qfrc_applied[j4_dof] = tau4

            mujoco.mj_step(model, data)

            # If using qfrc_applied fallback, clear it to avoid accumulation
            if not use_ctrl:
                data.qfrc_applied[j1_dof] = 0.0
                data.qfrc_applied[j2_dof] = 0.0
                data.qfrc_applied[j3_dof] = 0.0
                data.qfrc_applied[j4_dof] = 0.0

            # Next angles
            j1n = np.degrees(data.qpos[j1_qpos])
            j2n = np.degrees(data.qpos[j2_qpos])

            # Base velocities
            lin_v, ang_v = get_base_speeds()

            # Errors
            err1 = abs_angle_diff_deg(j1n, target1)
            err2 = abs_angle_diff_deg(j2n, target2)
            curr_err = err1 + err2

            # Terminal condition on total error
            done = curr_err < TOL_DEG

            # Reward components
            reward_progress = prev_err - curr_err
            base_penalty = -BASE_PENALTY_SCALE * (lin_v + ang_v)
            reward = reward_progress + STEP_PENALTY + base_penalty + (TERMINAL_BONUS if done else 0.0)

            # Next state with tiny-change suppression
            j1_eff = j1n if abs(j1n - j1) >= IGNORE_THRESHOLD_DEG else j1
            j2_eff = j2n if abs(j2n - j2) >= IGNORE_THRESHOLD_DEG else j2
            s_next_key = (angle_to_index(j1_eff), angle_to_index(j2_eff))

            # Q-update
            agent.update(s_key, a_idx, reward, s_next_key, done)
            total_reward += reward

            # Roll to next
            prev_err = curr_err
            j1, j2 = j1n, j2n

            # Render if viewer
            if viewer is not None:
                viewer.sync()
                if viewer_fps_sleep > 0:
                    time.sleep(viewer_fps_sleep)

            if done:
                break

        return total_reward, t + 1

    if with_viewer:
        # Visual training (slower)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for ep in range(episodes):
                total_reward, steps_used = _episode(viewer=viewer)
                if (ep + 1) % 10 == 0:
                    print(f"Episode {ep+1}/{episodes} | steps={steps_used} | R={total_reward:.2f} | eps={agent.eps:.3f}")
    else:
        # Headless training (faster)
        for ep in range(episodes):
            total_reward, steps_used = _episode(viewer=None)
            if (ep + 1) % 100 == 0:
                print(f"Episode {ep+1}/{episodes} | steps={steps_used} | R={total_reward:.2f} | eps={agent.eps:.3f}")

    with open(save_path, "wb") as f:
        pickle.dump(dict(agent.Q), f)
    print(f"Saved Q-table to {save_path}")
    return agent

# =========================
# Greedy rollout in viewer
# =========================
def greedy_rollout(agent, episodes=5, steps=500, fps_sleep=0.01):
    def greedy_action(s_key):
        qvals = [agent.get_q(s_key, idx) for idx in range(len(agent.actions))]
        a_idx = int(np.argmax(qvals))
        return agent.actions[a_idx]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(episodes):
            mujoco.mj_resetData(model, data)
            data.qpos[j1_qpos] = np.radians(random.uniform(*INIT_RANGE_DEG))
            data.qpos[j2_qpos] = np.radians(random.uniform(*INIT_RANGE_DEG))
            mujoco.mj_forward(model, data)

            for t in range(steps):
                j1 = np.degrees(data.qpos[j1_qpos])
                j2 = np.degrees(data.qpos[j2_qpos])
                s_key = (angle_to_index(j1), angle_to_index(j2))
                tau1, tau2, tau3, tau4 = greedy_action(s_key)

                if use_ctrl:
                    data.ctrl[a1] = tau1
                    data.ctrl[a2] = tau2
                    data.ctrl[a3] = tau3
                    data.ctrl[a4] = tau4
                else:
                    data.qfrc_applied[j1_dof] = tau1
                    data.qfrc_applied[j2_dof] = tau2
                    data.qfrc_applied[j3_dof] = tau3
                    data.qfrc_applied[j4_dof] = tau4

                mujoco.mj_step(model, data)
                if not use_ctrl:
                    data.qfrc_applied[j1_dof] = 0.0
                    data.qfrc_applied[j2_dof] = 0.0
                    data.qfrc_applied[j3_dof] = 0.0
                    data.qfrc_applied[j4_dof] = 0.0

                viewer.sync()
                if fps_sleep > 0:
                    time.sleep(fps_sleep)

if __name__ == "__main__":
    # Train headless for speed; switch with_viewer=True to observe learning live
    agent = train(episodes=EPISODES, steps=MAX_STEPS, save_path="q_table_nullspace.pkl", with_viewer=False)

    # Visualize greedy policy after training
    greedy_rollout(agent, episodes=3, steps=1000, fps_sleep=0.005)
