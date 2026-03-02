import os
import time
import numpy as np

# ---- matplotlib (safe) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mujoco
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ---- IMPORT YOUR ENV ----
from sac_code1 import (
    HybridFree10DOFEnv,
    XML_PATH,
    MAX_EPISODE_STEPS
)

# ================= SETTINGS =================
MODEL_PATH = "models_sac_free10dof/sac_free10dof_continued"
SAVE_DIR = "eval_plots"
SEED = 12
RENDER = True          # <<< turn OFF if not needed
FPS = 60


# =================================================
def make_env(render=False):
    def _init():
        env = HybridFree10DOFEnv(XML_PATH, seed=SEED, render=render)
        return Monitor(env)
    return _init


# =================================================
def smooth(x, w=10):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


# =================================================
def evaluate():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("🔍 Loading env...")
    env = DummyVecEnv([make_env(render=RENDER)])

    print("📦 Loading model...")
    model = SAC.load(MODEL_PATH, env=env)

    obs = env.reset()

    # ---------- LOGS ----------
    base_pos_log = []
    base_vel_log = []
    joint_pos_log = []
    joint_err_log = []
    reward_log = []

    print("▶ Running evaluation for FULL episode...")

    for step in range(MAX_EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)

        # ---- ACCESS MUJOCO STATE ----
        unwrapped = env.envs[0].unwrapped
        data = unwrapped.data

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        # ---- BASE ----
        base_pos_log.append(qpos[0:3])
        base_vel_log.append(qvel[0:3])

        # ---- JOINTS ----
        joint_pos = qpos[7:]
        joint_pos_log.append(joint_pos)
        joint_err_log.append(np.linalg.norm(joint_pos))

        reward_log.append(reward[0])

        if RENDER:
            time.sleep(1 / FPS)

    env.close()
    print("📊 Evaluation finished")

    save_plots(
        np.array(joint_pos_log),
        np.array(joint_err_log),
        np.array(base_pos_log),
        np.array(base_vel_log),
        np.array(reward_log),
    )


# =================================================
def save_plots(joint_pos, joint_err, base_pos, base_vel, rewards):
    t = np.arange(len(rewards))

    # ---- Joint Positions ----
    plt.figure(figsize=(8, 5))
    for i in range(joint_pos.shape[1]):
        y = smooth(joint_pos[:, i])
        plt.plot(t[-len(y):], y, label=f"q{i+1}")
    plt.legend()
    plt.title("Joint Positions")
    plt.xlabel("Timestep")
    plt.grid()
    plt.savefig(f"{SAVE_DIR}/joint_positions.png")
    plt.close()

    # ---- Joint Error ----
    y = smooth(joint_err)
    plt.figure()
    plt.plot(t[-len(y):], y)
    plt.title("Joint Error")
    plt.xlabel("Timestep")
    plt.grid()
    plt.savefig(f"{SAVE_DIR}/joint_error.png")
    plt.close()

    # ---- Base Position ----
    labels = ["x", "y", "z"]
    plt.figure()
    for i in range(3):
        y = smooth(base_pos[:, i])
        plt.plot(t[-len(y):], y, label=labels[i])
    plt.legend()
    plt.title("Base Position")
    plt.xlabel("Timestep")
    plt.grid()
    plt.savefig(f"{SAVE_DIR}/base_position.png")
    plt.close()

    # ---- Base Velocity ----
    plt.figure()
    for i in range(3):
        y = smooth(base_vel[:, i])
        plt.plot(t[-len(y):], y, label=labels[i])
    plt.legend()
    plt.title("Base Velocity")
    plt.xlabel("Timestep")
    plt.grid()
    plt.savefig(f"{SAVE_DIR}/base_velocity.png")
    plt.close()

    # ---- Reward ----
    y = smooth(rewards)
    plt.figure()
    plt.plot(t[-len(y):], y)
    plt.title("Reward")
    plt.xlabel("Timestep")
    plt.grid()
    plt.savefig(f"{SAVE_DIR}/reward.png")
    plt.close()

    print("✅ Saved plots in eval_plots/")
    print("   joint_positions.png")
    print("   joint_error.png")
    print("   base_position.png")
    print("   base_velocity.png")
    print("   reward.png")


# =================================================
if __name__ == "__main__":
    evaluate()

