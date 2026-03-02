from stable_baselines3 import PPO
from nullspace_env import NullspaceEnv

# Create environment
env = NullspaceEnv(model_path="nullspace.xml", steps=5000)

# Initialize RL agent
model = PPO("MlpPolicy", env, verbose=1)

# Train agent
model.learn(total_timesteps=200_000)

# Save trained model
model.save("ppo_nullspace")
