from stable_baselines3 import SAC
from nullspace_env import NullspaceEnv
import time

# --- Create env with render_mode=True ---
env = NullspaceEnv(render_mode=True)

# Load trained model
model = SAC.load("sac_nullspace_robot", env=env)

# Reset environment
obs, _ = env.reset()

# Run rollout with Mujoco viewer visible
for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    print(f"Step {step}: Reward: {reward}")

    # Slow down so you can see the motion in the viewer
    time.sleep(0.01)

    if done or truncated:
        obs, _ = env.reset()

env.close()

