import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from aanewcodeofsac import Hybrid5DOFEnv 

XML_PATH = "nullspace.xml"
MODELS_DIR = "sac_models1"
SEED = 0

# Create env
env = DummyVecEnv([
    lambda: Monitor(Hybrid5DOFEnv(xml_path=XML_PATH, seed=SEED))
])

# Load existing trained model
model = SAC.load(
    os.path.join(MODELS_DIR, "sac_hybrid_final"),
    env=env
)

# Reduce learning rate for safe fine-tuning
model.lr_schedule = lambda _: 1e-4

print("✅ Continuing training from checkpoint...")

# Continue training
model.learn(
    total_timesteps=400_000,              # extra steps to reach ~1M
    reset_num_timesteps=False
)

# Save updated model
model.save(os.path.join(MODELS_DIR, "sac_hybrid_1M"))

print("✅ Training extended to ~1M steps.")
