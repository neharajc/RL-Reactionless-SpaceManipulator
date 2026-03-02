import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from sac_code1 import HybridFree10DOFEnv, XML_PATH, SEED, MODELS_DIR, LOG_DIR
from sac_code1 import ProgressBarCallback, PeriodicRenderCallback, TOTAL_TIMESTEPS

# extra training steps
EXTRA_TIMESTEPS = 1_000_000  # e.g. add 10 lakh more

def make_env(seed):
    def _init():
        return Monitor(HybridFree10DOFEnv(XML_PATH, seed=seed, render=False))
    return _init

def continue_training():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    env = DummyVecEnv([make_env(SEED)])

    # load last trained model
    MODEL_PATH = os.path.join(MODELS_DIR, "sac_free10dof_final.zip")
    model = SAC.load(MODEL_PATH, env=env)

    callbacks = [
        ProgressBarCallback(TOTAL_TIMESTEPS + EXTRA_TIMESTEPS),
        PeriodicRenderCallback(render_every=100),
    ]

    model.learn(
        total_timesteps=EXTRA_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=False,   # <-- key: continue from previous step count
    )

    model.save(os.path.join(MODELS_DIR, "sac_free10dof_continued"))

if __name__ == "__main__":
    continue_training()

