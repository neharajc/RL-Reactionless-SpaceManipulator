import os
os.environ["MUJOCO_GL"] = "egl"   # Prevents Linux OpenGL crashes

import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

# Import your environment
from sactraining4 import Hybrid5DOFEnv

XML_PATH = "nullspace.xml"
MODEL_PATH = "sac_hybrid_final1.zip"   # <--- your final model file

# -------------------------------------------------------------
#                 LOAD ENV + MODEL
# -------------------------------------------------------------
env = Hybrid5DOFEnv(xml_path=XML_PATH, render=False, seed=123)

model = SAC.load(MODEL_PATH)
print("\n🚀 Loaded trained SAC model\n")

m = env.model
d = env.data

print("Opening MuJoCo Viewer...")

# -------------------------------------------------------------
#        START MUJOCO VIEWER (MOST STABLE MODE)
# -------------------------------------------------------------
with mujoco.viewer.launch(m, d) as viewer:
    for ep in range(3):

        print(f"\n================ EPISODE {ep} ================\n")

        obs, _ = env.reset()
        total_reward = 0

        for step in range(600):

            # SAC prediction
            action, _ = model.predict(obs, deterministic=True)

            # Step env
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Update the viewer screen
            viewer.sync()

            # Slow down so you can SEE the motion
            time.sleep(0.02)

            print(f"Step {step:03d}  | Reward={reward:.3f}")

            if terminated or truncated:
                break

        print(f"Episode reward = {total_reward:.2f}")

print("\nEvaluation Finished.\n")

