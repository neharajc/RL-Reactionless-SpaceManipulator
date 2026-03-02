import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# Import your environment class
from aanewcodeofsac import Hybrid5DOFEnv

XML_PATH = "nullspace.xml"
MODEL_PATH = "sac_models1/sac_hybrid_final.zip"

# Create env with rendering ON
env = Monitor(Hybrid5DOFEnv(xml_path=XML_PATH, render=True))

# Load trained model
model = SAC.load(MODEL_PATH)

print("✅ Model loaded. Starting test...")

# Run multiple test episodes
for ep in range(5):
    print(f"\n=== Episode {ep+1} ===")
    obs, _ = env.reset()  # random positions from your reset()

    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        # Print joint positions
        if step % 20 == 0:
            qpos = env.unwrapped.data.qpos.copy()
            print(f"Step {step:03d} | qpos (rad): {qpos}")

        # Check if reached near zero
        joint_error = np.sum(qpos[1:] ** 2)
        if joint_error < 0.01:
            print("✅ Target reached (near zero angles)")
            break

        if terminated or truncated:
            break

env.close()
