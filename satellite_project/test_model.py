from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# ✅ Import environment + XML path from your training file
from aanewcodeofsac import Hybrid5DOFEnv, XML_PATH

print("Loading environment and model...")

# Create environment with rendering
env = Monitor(Hybrid5DOFEnv(xml_path=XML_PATH, render=True))

# Load trained model
model = SAC.load("sac_models1/sac_hybrid_final", env=env)

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

