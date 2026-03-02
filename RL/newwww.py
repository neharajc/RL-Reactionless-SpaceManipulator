"""
Tabular Q-learning for 4-joint planar manipulator (MuJoCo).
Follows your spec:
 - 0.5 degree discretization (720 bins per joint over 360°)
 - 81 discrete torque actions: each joint torque ∈ {-1,0,1} (3^4)
 - Tiny-change suppression: threshold 0.05° -> reuse previous angle
 - Base reaction penalty using |qdot_base|
 - Reward = (prev_err - curr_err) - 0.01 + base_penalty, plus +100 terminal bonus
 - Success when |j1| + |j2| < 0.5° (in radians)
"""

import mujoco
import mujoco.viewer
import numpy as np
import pickle
import math
import random
from collections import defaultdict
from itertools import product
import os
import time

# -------------------------
# Config / Hyperparams
# -------------------------
MODEL_PATH = "newnull.xml"                # replace with your model file
ACTUATOR_NAMES = ["joint1_motor", "joint2_motor", "joint3_motor", "joint4_motor"]
# If your actuators are named differently, set appropriately; we'll map by name to ctrl indices.

BASE_JOINT_NAME = "base_joint"          # hinge base joint name (in the MuJoCo model)
JOINT1_NAME = "joint1"
JOINT2_NAME = "joint2"

# discretization
DEG_PER_BIN = 0.5
BINS_PER_REV = int(360 / DEG_PER_BIN)  # 720
ANGLE_EPS_DEG = 0.05                    # tiny-change suppression threshold in degrees
ANGLE_EPS = math.radians(ANGLE_EPS_DEG) # in radians

# actions: per-joint torque levels
TORQUE_LEVELS = [-1.0, 0.0, 1.0]        # baseline torque levels; actuator ctrlrange should match scaled units
# will generate 3^4 = 81 actions
ACTION_LIST = list(product(TORQUE_LEVELS, repeat=4))
N_ACTIONS = len(ACTION_LIST)

# Q-learning params
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_FRAC = 0.60   # reach EPS_END at 60% of total_episodes via exponential schedule

# episodes / steps
TOTAL_EPISODES = 20000
STEPS_PER_EP = 750

# rewards
STEP_PENALTY = -0.01
TERMINAL_BONUS = 100.0
BASE_PENALTY_MULT = -0.1  # multiply by |qdot_base|

# success threshold
SUCCESS_THRESH_DEG = 0.5
SUCCESS_THRESH = math.radians(SUCCESS_THRESH_DEG)

# Q-table persistence
Q_SAVE_PATH = "q_table.pkl"

# miscellaneous
SEED = 0
RNG = random.Random(SEED)
NP_RNG = np.random.default_rng(SEED)


# -------------------------
# Utilities
# -------------------------
def angle_wrap_to_2pi(x):
    """Return angle in [0, 2π)."""
    return x % (2 * math.pi)

def shortest_angular_distance(a, b):
    """Return shortest signed angular distance from a to b (radians)."""
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    return diff

def angular_abs_distance(a, b):
    """Return absolute shortest angular distance between angles a and b."""
    return abs(shortest_angular_distance(a, b))

def angle_to_bin(angle_rad):
    """
    Convert angle in radians (any range) to bin index [0, BINS_PER_REV-1].
    We map angle to [0,2π) then to bins.
    """
    a = angle_wrap_to_2pi(angle_rad)
    # bin width in radians
    bin_w = (2 * math.pi) / BINS_PER_REV
    idx = int(a // bin_w)
    # safety clamp
    return idx % BINS_PER_REV

def compute_total_error(j1, j2):
    """Total error for target 0 for joints 1 and 2 using shortest angular distance."""
    return angular_abs_distance(j1, 0.0) + angular_abs_distance(j2, 0.0)


# -------------------------
# MuJoCo model/data helpers
# -------------------------
def map_names_to_indices(model):
    """Return dictionaries mapping joint/actuator names to indices and DOF addresses needed."""
    name_to_act_idx = {}
    for i in range(model.nu):  # control indices
        name = model.actuator(i).name.decode() if isinstance(model.actuator(i).name, bytes) else model.actuator(i).name
        name_to_act_idx[name] = i

    name_to_joint_qposadr = {}
    name_to_joint_dofadr = {}
    for jid in range(model.njnt):
        jname = model.joint(jid).name.decode() if isinstance(model.joint(jid).name, bytes) else model.joint(jid).name
        qposadr = model.jnt_qposadr[jid]
        dofadr = model.jnt_dofadr[jid]
        name_to_joint_qposadr[jname] = qposadr
        name_to_joint_dofadr[jname] = dofadr

    return name_to_act_idx, name_to_joint_qposadr, name_to_joint_dofadr

# -------------------------
# Q-Learner (tabular, sparse dict)
# -------------------------
class QLearnerSparse:
    def __init__(self, alpha=ALPHA, gamma=GAMMA):
        self.alpha = alpha
        self.gamma = gamma
        # Use defaultdict(float) to implicitly initialize Q to 0.0
        self.Q = defaultdict(float)

    def key(self, bin_j1, bin_j2, action_idx):
        return (bin_j1, bin_j2, action_idx)

    def get_q(self, bin_j1, bin_j2, action_idx):
        return self.Q[self.key(bin_j1, bin_j2, action_idx)]

    def update(self, bin_j1, bin_j2, action_idx, reward, next_bin_j1, next_bin_j2):
        # Standard Q-learning update
        s_key = self.key(bin_j1, bin_j2, action_idx)
        q = self.Q[s_key]
        # compute max_a' Q(s', a')
        max_next_q = max((self.Q[(next_bin_j1, next_bin_j2, a)] for a in range(N_ACTIONS)), default=0.0)
        td = reward + self.gamma * max_next_q - q
        self.Q[s_key] = q + self.alpha * td

    def best_action_and_value(self, bin_j1, bin_j2):
        # return (action_idx, q_value) with random tie-breaking
        qvals = [self.Q[(bin_j1, bin_j2, a)] for a in range(N_ACTIONS)]
        max_q = max(qvals)
        max_idxs = [i for i, v in enumerate(qvals) if v == max_q]
        chosen = NP_RNG.choice(max_idxs)
        return chosen, max_q

    def save(self, path=Q_SAVE_PATH):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path=Q_SAVE_PATH):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.Q = defaultdict(float, d)


# -------------------------
# Epsilon scheduling (exponential to EPS_END at fraction EPS_DECAY_FRAC)
# -------------------------
def epsilon_for_episode(ep_idx, total_episodes=TOTAL_EPISODES):
    if total_episodes <= 0:
        return EPS_END
    # want eps(t) = EPS_END + (EPS_START - EPS_END) * exp(-k * t)
    # solve for k so that at t = EPS_DECAY_FRAC * total_episodes, eps = EPS_END + 0.01*(EPS_START - EPS_END) (approx)
    # Simpler: compute decay such that at t_target it equals EPS_END (exact) -> k = -ln(EPS_END/EPS_START)/t_target
    t_target = max(1, int(EPS_DECAY_FRAC * total_episodes))
    if EPS_START <= EPS_END:
        return EPS_END
    k = -math.log(EPS_END / EPS_START) / t_target
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-k * ep_idx)
    return max(EPS_END, eps)


# -------------------------
# Main training loop
# -------------------------
def train():
    # load model & data
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Map names -> indices
    act_name_to_idx, joint_to_qposadr, joint_to_dofadr = map_names_to_indices(model)

    # Build actuator ctrl indices list in same order as ACTION_LIST mapping (joint1..4)
    ctrl_indices = []
    for aname in ACTUATOR_NAMES:
        if aname not in act_name_to_idx:
            raise RuntimeError(f"Actuator named '{aname}' not found in model actuators.")
        ctrl_indices.append(act_name_to_idx[aname])

    # Base joint indices
    if BASE_JOINT_NAME not in joint_to_dofadr:
        raise RuntimeError(f"Base joint name '{BASE_JOINT_NAME}' not found in model joints.")
    base_dofadr = joint_to_dofadr[BASE_JOINT_NAME]

    # Joint qpos indices for joint1 and joint2
    for jn in (JOINT1_NAME, JOINT2_NAME):
        if jn not in joint_to_qposadr:
            raise RuntimeError(f"Joint name '{jn}' not found in model joints.")

    j1_qposadr = joint_to_qposadr[JOINT1_NAME]
    j2_qposadr = joint_to_qposadr[JOINT2_NAME]
    j1_dofadr = joint_to_dofadr[JOINT1_NAME]
    j2_dofadr = joint_to_dofadr[JOINT2_NAME]

    # instantiate Q-learner
    qlearner = QLearnerSparse()

    # initialize previous angle memory for tiny-change suppression
    prev_angles = {JOINT1_NAME: None, JOINT2_NAME: None}

    # helper to initialize random starting angles uniformly in [0°, 10°] for j1 and j2 (in radians)
    def random_init_angles():
        deg = NP_RNG.uniform(0.0, 10.0)
        deg2 = NP_RNG.uniform(0.0, 10.0)
        return math.radians(deg), math.radians(deg2)

    # for progress logging
    last_save_time = time.time()
    episodes_since_save = 0

    for ep in range(TOTAL_EPISODES):
        # epsilon
        eps = epsilon_for_episode(ep, TOTAL_EPISODES)

        # initialize qpos for episode
        # zero everything first
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0

        # set random initial angles for j1, j2 in qpos
        a1, a2 = random_init_angles()
        data.qpos[j1_qposadr] = a1
        data.qpos[j2_qposadr] = a2
        mujoco.mj_forward(model, data)  # propagate

        # set prev_angles to the initial measurement
        prev_angles[JOINT1_NAME] = data.qpos[j1_qposadr]
        prev_angles[JOINT2_NAME] = data.qpos[j2_qposadr]

        # compute initial discretized state
        # apply tiny-change suppression at the first step is not meaningful since no prev; we saved initial angles
        bin_j1 = angle_to_bin(prev_angles[JOINT1_NAME])
        bin_j2 = angle_to_bin(prev_angles[JOINT2_NAME])

        # initial error
        prev_err = compute_total_error(prev_angles[JOINT1_NAME], prev_angles[JOINT2_NAME])

        done = False
        for step in range(STEPS_PER_EP):
            # choose action: eps-greedy
            if NP_RNG.random() < eps:
                action_idx = NP_RNG.integers(0, N_ACTIONS)
            else:
                action_idx, _ = qlearner.best_action_and_value(bin_j1, bin_j2)

            # apply torques to data.ctrl using mapping ctrl_indices
            torques = ACTION_LIST[action_idx]
            # Zero ctrl array then set the indexed actuators
            data.ctrl[:] = 0.0
            for i, ctrl_idx in enumerate(ctrl_indices):
                data.ctrl[ctrl_idx] = torques[i]

            # step simulation
            mujoco.mj_step(model, data)

            # read new joint angles and base velocity using jnt_qposadr/jnt_dofadr
            raw_j1 = data.qpos[j1_qposadr]
            raw_j2 = data.qpos[j2_qposadr]

            # tiny-change suppression: if change < ANGLE_EPS, reuse previous angle value
            if prev_angles[JOINT1_NAME] is not None and abs(shortest_angular_distance(raw_j1, prev_angles[JOINT1_NAME])) < ANGLE_EPS:
                j1 = prev_angles[JOINT1_NAME]
            else:
                j1 = raw_j1

            if prev_angles[JOINT2_NAME] is not None and abs(shortest_angular_distance(raw_j2, prev_angles[JOINT2_NAME])) < ANGLE_EPS:
                j2 = prev_angles[JOINT2_NAME]
            else:
                j2 = raw_j2

            # update prev_angles for next step
            prev_angles[JOINT1_NAME] = j1
            prev_angles[JOINT2_NAME] = j2

            # discretize
            next_bin_j1 = angle_to_bin(j1)
            next_bin_j2 = angle_to_bin(j2)

            # compute reward
            curr_err = compute_total_error(j1, j2)
            delta_err = prev_err - curr_err
            base_omega = abs(data.qvel[base_dofadr])  # magnitude of base hinge angular speed
            base_penalty = BASE_PENALTY_MULT * base_omega
            reward = delta_err + STEP_PENALTY + base_penalty

            done = False
            if curr_err < SUCCESS_THRESH:
                reward += TERMINAL_BONUS
                done = True

            # Q update
            qlearner.update(bin_j1, bin_j2, action_idx, reward, next_bin_j1, next_bin_j2)

            # advance for next step
            bin_j1, bin_j2 = next_bin_j1, next_bin_j2
            prev_err = curr_err

            if done:
                break

        # optional periodic saves/logging
        episodes_since_save += 1
        if (time.time() - last_save_time) > 60 or episodes_since_save >= 500:
            qlearner.save(Q_SAVE_PATH)
            last_save_time = time.time()
            episodes_since_save = 0
            print(f"[ep {ep+1}/{TOTAL_EPISODES}] saved Q-table. eps={eps:.4f}")

    # final save
    qlearner.save(Q_SAVE_PATH)
    print("Training complete. Q-table saved to", Q_SAVE_PATH)


# -------------------------
# Inference/Playback
# -------------------------
def run_inference(render=True, max_steps=STEPS_PER_EP):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    act_name_to_idx, joint_to_qposadr, joint_to_dofadr = map_names_to_indices(model)

    ctrl_indices = [act_name_to_idx[n] for n in ACTUATOR_NAMES]
    j1_qposadr = joint_to_qposadr[JOINT1_NAME]
    j2_qposadr = joint_to_qposadr[JOINT2_NAME]
    base_dofadr = joint_to_dofadr[BASE_JOINT_NAME]

    qlearner = QLearnerSparse()
    qlearner.load(Q_SAVE_PATH)
    print("Loaded Q-table with", len(qlearner.Q), "entries.")

    # viewer if requested
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(model, data)  # or mujoco.viewer.launch(model, data) depending on version

    # initialize from a few test starts
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    # pick a test initialization in [0,10] deg again
    data.qpos[j1_qposadr] = math.radians(NP_RNG.uniform(0, 10))
    data.qpos[j2_qposadr] = math.radians(NP_RNG.uniform(0, 10))
    mujoco.mj_forward(model, data)

    prev_angles = {JOINT1_NAME: data.qpos[j1_qposadr], JOINT2_NAME: data.qpos[j2_qposadr]}
    bin_j1 = angle_to_bin(prev_angles[JOINT1_NAME])
    bin_j2 = angle_to_bin(prev_angles[JOINT2_NAME])

    for step in range(max_steps):
        # greedy action
        action_idx, _ = qlearner.best_action_and_value(bin_j1, bin_j2)
        torques = ACTION_LIST[action_idx]
        data.ctrl[:] = 0.0
        for i, ctrl_idx in enumerate(ctrl_indices):
            data.ctrl[ctrl_idx] = torques[i]

        mujoco.mj_step(model, data)

        # read angles
        raw_j1 = data.qpos[j1_qposadr]
        raw_j2 = data.qpos[j2_qposadr]

        # tiny-change suppression
        if abs(shortest_angular_distance(raw_j1, prev_angles[JOINT1_NAME])) < ANGLE_EPS:
            j1 = prev_angles[JOINT1_NAME]
        else:
            j1 = raw_j1
        if abs(shortest_angular_distance(raw_j2, prev_angles[JOINT2_NAME])) < ANGLE_EPS:
            j2 = prev_angles[JOINT2_NAME]
        else:
            j2 = raw_j2

        prev_angles[JOINT1_NAME] = j1
        prev_angles[JOINT2_NAME] = j2

        bin_j1 = angle_to_bin(j1)
        bin_j2 = angle_to_bin(j2)

        total_err = compute_total_error(j1, j2)
        if render and viewer is not None:
            viewer.sync()

        if total_err < SUCCESS_THRESH:
            print(f"Success at step {step}. total_err (deg) = {math.degrees(total_err):.4f}")
            break

    if render and viewer is not None:
        viewer.close()

# -------------------------
# Utility: quick check of actuator->joint mapping & ctrlrange
# -------------------------
def check_model_consistency():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    act_name_to_idx, joint_to_qposadr, joint_to_dofadr = map_names_to_indices(model)
    # print actuator mapping and ctrlrange / gear
    print("Actuator info (name -> ctrl idx, ctrlrange, gear):")
    for i in range(model.nu):
        a = model.actuator(i)
        name = a.name.decode() if isinstance(a.name, bytes) else a.name
        print(f"  {name}: ctrl_idx={i}, ctrlrange={a.ctrlrange}, gear={a.gear}")

    # print joint mapping
    print("\nJoints (name -> qposadr, dofadr):")
    for jid in range(model.njnt):
        j = model.joint(jid)
        name = j.name.decode() if isinstance(j.name, bytes) else j.name
        print(f"  {name}: qposadr={model.jnt_qposadr[jid]}, dofadr={model.jnt_dofadr[jid]}")

# -------------------------
# Entry point helpers
# -------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        check_model_consistency()
    elif cmd == "train":
        train()
    elif cmd == "run":
        run_inference(render=True)

