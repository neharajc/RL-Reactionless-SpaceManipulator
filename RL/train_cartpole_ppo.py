import gymnasium as gym
from stable_baselines3 import PPO
import os

env_id = "CartPole-v1"
logdir = "logs/cartpole_ppo"
os.makedirs(logdir, exist_ok=True)  # make sure folder exists

env = gym.make(env_id)

# Tell SB3 to log training data into TensorBoard
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir   # <--- critical line
)

# Train the agent
model.learn(total_timesteps=50_000)

# Save the trained agent
model.save(os.path.join(logdir, "ppo_cartpole"))

