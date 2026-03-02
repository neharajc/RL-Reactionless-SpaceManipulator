# qlearn_nullspace.py
import argparse
import time
import math
import random
import pickle
from collections import defaultdict
from typing import Optional, Tuple, List

import numpy as np
import mujoco
import mujoco.viewer

# -------------------------
# Config (matches your requirements)
# -------------------------
DEG_BIN = 0.5                    # 360/0.5 = 720 bins
TOL_DEG = 0.5                    # terminal tolerance on total error (j1+j2)
IGNORE_THRESHOLD_DEG = 0.05      # ignore tiny angle changes for stable bins

EPISODES = 20000                 # scale up for coverage
MAX_STEPS = 750                  # longer horizon
ALPHA = 0.2                      # learning rate
GAMMA = 0.995                    # discount
EPS_START = 1.0                  # initial ε
EPS_MIN = 0.01                   # ε floor
EPS_DECAY_FRAC = 0.6             # near-min by ~60% of training

STEP_PENALTY = -0.01             # small negative per-step
BASE_PENALTY_SCALE = 0.1         # reactionless penalty weight

INIT_RANGE_DEG = (0.0, 10.0)     # random initial j1,j2 in [0,10] deg
TARGET1 = 0.0                    # fixed targets (deg)
TARGET2 = 0.0                    # fixed targets (deg)

# torque levels; denser set helps with finer discretization
TORQUE_LEVELS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
ACTIONS = [(t1, t2, t3, t4)
           for t1 in TORQUE_LEVELS
           for t2 in TORQUE_LEVELS
           for t3 in TORQUE_LEVELS
           for t4 in TORQUE_LEVELS]   # 7^4 = 2401 actions

# -------------------------
# Angle helpers
# -------------------------
def norm_angle_deg(a: float) -> float:
    a = a % 360.0
    return a + 360.0 if a < 0 else a

def angle_to_index(angle_deg: float) -> int:
    a = norm_angle_deg(angle_deg)
    idx = int(a // DEG_BIN)
    max_bins = int(round(360.0 / DEG_BIN))
    return min(idx, max_bins - 1)

def abs_angle_diff_deg(a: float, b: float) -> float:
    d = abs(norm_angle_deg(a) - norm_angle_deg(b))
    return min(d, 360.0 - d)

def epsilon_by_episode(ep: int, episodes=EPISODES,
                       eps_start=EPS_START, eps_min=EPS_MIN, decay_frac=EPS_DECAY_FRAC) -> float:
    target_ep = max(1, int(decay_frac * episodes))
    k = math.log(max(eps_start / eps_min, 1.0001)) / target_ep
    eps = eps_start * math.exp(-k * ep)
    return max(eps_min, eps)

# -------------------------
# Q-learner (sparse dict)
# -------------------------
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
        qvals = [self.get_q(s_key, i) for i in range(len(self.actions))]
        maxq = max(qvals)
        # random tie-break
        cands = [i for i, v in enumerate(qvals) if v == maxq]
        return random.choice(cands)

    def update(self, s_key, a_idx, reward, s_next_key, done):
        q_sa = self.get_q(s_key, a_idx)
        target = reward if done else reward + self.gamma * max(self.get_q(s_next_key, i) for i in range(len(self.actions)))
        self.Q[(s_key, a_idx)] = q_sa + self.alpha * (target - q_sa)

# -------------------------
# MuJoCo wiring
# -------------------------
def jid(model, name: str) -> int:
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
        raise RuntimeError(f"Joint '{name}' not found in XML")
    return j

def actuator_for_joint(model, jid: int) -> Optional[int]:
    # map actuator transmission to the joint id
    for a in range(model.nu):
        if int(model.actuator_trnid[a][0]) == jid:
            return a
    return None

def build_model_data(xml_path: str):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Resolve joints (expect joint1..joint4 and base_joint)
    base_id = jid(model, "base_joint")
    j1_id = jid(model, "joint1")
    j2_id = jid(model, "joint2")
    j3_id = jid(model, "joint3")
    j4_id = jid(model, "joint4")

    # qpos addresses for angles; dof addresses for qvel/forces
    j1_qpos = model.jnt_qposadr[j1_id]
    j2_qpos = model.jnt_qposadr[j2_id]
    j3_qpos = model.jnt_qposadr[j3_id]
    j4_qpos = model.jnt_qposadr[j4_id]
    base_dof = model.jnt_dofadr[base_id]

    # actuators for each joint: must be <motor> for torque control
    a1 = actuator_for_joint(model, j1_id)
    a2 = actuator_for_joint(model, j2_id)
    a3 = actuator_for_joint(model, j3_id)
    a4 = actuator_for_joint(model, j4_id)
    if not all(a is not None for a in (a1, a2, a3, a4)):
        raise RuntimeError("Expected <motor> actuators for joint1..joint4; update XML to torque motors with proper transmission.")  # noqa

    return (model, data,
            {'base_id': base_id, 'j1_id': j1_id, 'j2_id': j2_id, 'j3_id': j3_id, 'j4_id': j4_id},
            {'j1_qpos': j1_qpos, 'j2_qpos': j2_qpos, 'j3_qpos': j3_qpos, 'j4_qpos': j4_qpos, 'base_dof': base_dof},
            {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4})

# -------------------------
# Training
# -------------------------
def train(xml: str, episodes=EPISODES, steps=MAX_STEPS,
          save_path="q_table.pkl", with_viewer=False, realtime=True, seed=123):
    # Load model and wiring
    model, data, jids, addrs, acts = build_model_data(xml)
    rng = np.random.default_rng(seed)
    agent = QLearnerSparse(ACTIONS)

    # Convenience handles
    j1_qpos, j2_qpos = addrs['j1_qpos'], addrs['j2_qpos']
    base_dof = addrs['base_dof']
    a1, a2, a3, a4 = acts['a1'], acts['a2'], acts['a3'], acts['a4']

    def run_episode(ep_idx: int, viewer=None):
        mujoco.mj_resetData(model, data)

        # Randomize initial angles for j1, j2 in [0,10] deg (model uses radians)
        data.qpos[j1_qpos] = np.radians(rng.uniform(*INIT_RANGE_DEG))
        data.qpos[j2_qpos] = np.radians(rng.uniform(*INIT_RANGE_DEG))
        mujoco.mj_forward(model, data)

        # Initial state
        j1 = float(np.degrees(data.qpos[j1_qpos]))
        j2 = float(np.degrees(data.qpos[j2_qpos]))
        prev_err = abs_angle_diff_deg(j1, TARGET1) + abs_angle_diff_deg(j2, TARGET2)
        ep_ret = 0.0

        for t in range(steps):
            # ε schedule
            agent.eps = epsilon_by_episode(ep_idx)

            # Discrete state from 0.5° bins (360/0.5)
            s_key = (angle_to_index(j1), angle_to_index(j2))

            # Action: 4-joint torques (j3,j4 for nullspace compensation)
            a_idx = agent.choose_action(s_key)
            tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

            # Apply torques via motor actuators
            data.ctrl[a1] = tau1
            data.ctrl[a2] = tau2
            data.ctrl[a3] = tau3
            data.ctrl[a4] = tau4

            # Step physics
            step_start = time.time()
            mujoco.mj_step(model, data)

            # Read next angles
            j1n = float(np.degrees(data.qpos[j1_qpos]))
            j2n = float(np.degrees(data.qpos[j2_qpos]))

            # Base hinge angular speed magnitude for reactionless penalty
            base_ang_speed = abs(float(data.qvel[base_dof]))
            base_lin_speed = 0.0  # hinge base has no linear dofs

            # Errors and termination
            err1 = abs_angle_diff_deg(j1n, TARGET1)
            err2 = abs_angle_diff_deg(j2n, TARGET2)
            curr_err = err1 + err2
            done = curr_err < TOL_DEG

            # Reward: progress + step penalty + base penalty + terminal bonus
            reward_progress = prev_err - curr_err
            base_penalty = -BASE_PENALTY_SCALE * (base_lin_speed + base_ang_speed)
            reward = reward_progress + STEP_PENALTY + base_penalty + (100.0 if done else 0.0)

            # Tiny-change suppression to stabilize bins
            j1_eff = j1n if abs(j1n - j1) >= IGNORE_THRESHOLD_DEG else j1
            j2_eff = j2n if abs(j2n - j2) >= IGNORE_THRESHOLD_DEG else j2
            s_next = (angle_to_index(j1_eff), angle_to_index(j2_eff))

            # Q update
            agent.update(s_key, a_idx, reward, s_next, done)
            ep_ret += reward
            prev_err = curr_err
            j1, j2 = j1n, j2n

            # Viewer sync and optional real-time pacing
            if viewer is not None:
                viewer.sync()
                if realtime:
                    dt = model.opt.timestep - (time.time() - step_start)
                    if dt > 0:
                        time.sleep(dt)

            if done:
                break

        return ep_ret, t + 1, model, data

    if with_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for ep in range(episodes):
                R, steps_used, _, _ = run_episode(ep, viewer=viewer)
                if (ep + 1) % 10 == 0:
                    print(f"[train] ep={ep+1}/{episodes} steps={steps_used} R={R:.2f} eps={agent.eps:.3f}", flush=True)
    else:
        for ep in range(episodes):
            R, steps_used, _, _ = run_episode(ep, viewer=None)
            if (ep + 1) % 100 == 0:
                print(f"[train] ep={ep+1}/{episodes} steps={steps_used} R={R:.2f} eps={agent.eps:.3f}", flush=True)

    with open(save_path, "wb") as f:
        pickle.dump(dict(agent.Q), f)
    print(f"Saved Q-table to {save_path}", flush=True)
    return agent, model, data, addrs, acts

# -------------------------
# Greedy rollout with viewer
# -------------------------
def greedy_rollout(agent, model, data, addrs, acts, episodes=2, steps=1200, realtime=True):
    j1_qpos, j2_qpos = addrs['j1_qpos'], addrs['j2_qpos']
    a1, a2, a3, a4 = acts['a1'], acts['a2'], acts['a3'], acts['a4']

    def greedy_action(s_key):
        qvals = [agent.get_q(s_key, i) for i in range(len(agent.actions))]
        a_idx = int(np.argmax(qvals))
        return agent.actions[a_idx]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(episodes):
            mujoco.mj_resetData(model, data)
            # randomize start
            data.qpos[j1_qpos] = np.radians(random.uniform(*INIT_RANGE_DEG))
            data.qpos[j2_qpos] = np.radians(random.uniform(*INIT_RANGE_DEG))
            mujoco.mj_forward(model, data)

            for t in range(steps):
                j1 = float(np.degrees(data.qpos[j1_qpos]))
                j2 = float(np.degrees(data.qpos[j2_qpos]))
                s_key = (angle_to_index(j1), angle_to_index(j2))

                tau1, tau2, tau3, tau4 = greedy_action(s_key)
                data.ctrl[a1] = tau1
                data.ctrl[a2] = tau2
                data.ctrl[a3] = tau3
                data.ctrl[a4] = tau4

                tick = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()
                if realtime:
                    dt = model.opt.timestep - (time.time() - tick)
                    if dt > 0:
                        time.sleep(dt)

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="newnull.xml", help="Path to MJCF (e.g., newnull.xml)")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--viewer", action="store_true", help="Render training live")
    args = parser.parse_args()

    agent, model, data, addrs, acts = train(
        xml=args.xml,
        episodes=args.episodes,
        steps=args.steps,
        save_path="q_table_nullspace.pkl",
        with_viewer=args.viewer,
        realtime=True,
        seed=123
    )

    # Visualize greedy policy after training
    greedy_rollout(agent, model, data, addrs, acts, episodes=2, steps=1200, realtime=True)
