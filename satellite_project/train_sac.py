from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from nullspace_env import NullspaceEnv

# --- Environment wrapper for SB3 ---
def make_env():
    return NullspaceEnv(render_mode=True)  # set True to see simulation

env = DummyVecEnv([make_env])

# --- Create SAC model ---
model = SAC("MlpPolicy", env, verbose=1)

# --- Train the model ---
# Increase timesteps for better convergence
model.learn(total_timesteps=200000)

# --- Save the trained model ---
model.save("sac_nullspace_robot")

env.close()

