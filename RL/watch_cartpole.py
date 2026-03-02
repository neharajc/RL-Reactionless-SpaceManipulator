import gymnasium as gym
from stable_baselines3 import PPO
import time

# Load the trained model (adjust path if needed)
model = PPO.load("logs/cartpole_ppo/ppo_cartpole")

# Create environment with rendering enabled
env = gym.make("CartPole-v1", render_mode="human")

obs, _ = env.reset()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
    time.sleep(0.02)  # slow down so you can watch
env.close()
