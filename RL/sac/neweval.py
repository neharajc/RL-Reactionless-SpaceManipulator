import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

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
    qerror_log = []

    for step in range(MAX_EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        unwrapped_env = env.envs[0].unwrapped
        data = unwrapped_env.data
        
        # EXACTLY like target: joints 1,2,3,4 error
        joint_q = data.qpos[1:5]      # joints 1..4  
        error = np.zeros(4) - joint_q # target = 0
        qerror_log.append(error)

        if dones[0]:
            break

    qerror_log = np.array(qerror_log)

    # ---- MATCH TARGET GRAPH STYLE ----
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green', 'red']  # Match target colors
    
    for i in range(4):
        plt.plot(qerror_log[:, i], color=colors[i], linewidth=2.5)
    
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Error (rad)", fontsize=12)
    plt.title("Joint Error (Target - Actual)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # EXACT LEGEND MATCHING TARGET
    plt.legend(['Error[1]', 'Error[2]', 'Error[3]', 'Error[4]'], 
               bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("target_qerror_style.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Saved target_qerror_style.png")

if __name__ == "__main__":
    evaluate_and_plot()

