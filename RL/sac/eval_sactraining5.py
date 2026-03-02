import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Import your environment
from sactraining5 import Hybrid5DOFEnv, XML_PATH, MAX_EPISODE_STEPS

MODEL_PATH = "sac_ideal_final.zip"
SEED = 100

def make_env():
    def _init():
        return Monitor(Hybrid5DOFEnv(XML_PATH, seed=SEED))
    return _init

def evaluate_and_plot():
    env = DummyVecEnv([make_env()])
    model = SAC.load(MODEL_PATH, env=env)

    obs = env.reset()

    qpos_log = []
    qerror_log = []  # New log for error
    base_vel_log = []
    reward_log = []

    for step in range(MAX_EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        # Access the simulator data
        unwrapped_env = env.envs[0].unwrapped
        data = unwrapped_env.data

        # --- CALCULATE JOINT ERROR ---
        # Assuming target is 0. If your env has a 'goal', use: target = unwrapped_env.goal
        target_qpos = np.zeros_like(data.qpos) 
        current_qpos = data.qpos.copy()
        error = target_qpos - current_qpos

        # Log values
        qpos_log.append(current_qpos)
        qerror_log.append(error)
        base_vel_log.append(data.qvel[0])
        reward_log.append(rewards[0])

        if dones[0]:
            print(f"Episode finished at step {step}")
            break

    # Convert to numpy arrays
    qpos_log = np.array(qpos_log)
    qerror_log = np.array(qerror_log)
    base_vel_log = np.array(base_vel_log)
    reward_log = np.array(reward_log)

    # --- TRIM FINAL TIMESTEP ---
    if len(reward_log) > 1:
        qpos_log = qpos_log[:-1]
        qerror_log = qerror_log[:-1]
        base_vel_log = base_vel_log[:-1]
        reward_log = reward_log[:-1]

    plot_results(qpos_log, qerror_log, base_vel_log, reward_log)

def plot_results(qpos, qerror, base_vel, rewards):
    t = np.arange(len(rewards))

    # ---- 1. Joint Positions ----
    plt.figure(figsize=(8, 5))
    for i in range(qpos.shape[1]):
        plt.plot(t, qpos[:, i], label=f"qpos[{i}]")
    plt.xlabel("Timestep")
    plt.ylabel("Position (rad)")
    plt.title("Joint Positions")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("qpos_plot_clean.png")
    plt.close()

    # ---- 2. Joint Error ----
    plt.figure(figsize=(8, 5))
    for i in range(qerror.shape[1]):
        plt.plot(t, qerror[:, i], label=f"Error[{i}]")
    plt.axhline(0, color='black', linestyle='--', alpha=0.3) # Zero error line
    plt.xlabel("Timestep")
    plt.ylabel("Error (rad)")
    plt.title("Joint Error (Target - Actual)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("qerror_plot_clean.png")
    plt.close()

    # ---- 3. Base Velocity ----
    plt.figure(figsize=(8, 5))
    plt.plot(t, base_vel, color='tab:blue', linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Base Velocity")
    plt.title("Base Velocity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("base_vel_plot_clean.png")
    plt.close()

    # ---- 4. Reward ----
    plt.figure(figsize=(8, 5))
    plt.plot(t, rewards, color='tab:green', linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward per Step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_plot_clean.png")
    plt.close()

    print("✅ All plots saved, including qerror_plot_clean.png")

if __name__ == "__main__":
    evaluate_and_plot()
