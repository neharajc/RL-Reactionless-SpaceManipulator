import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.evaluation import evaluate_policy


# ================= USER SETTINGS =================
XML_PATH = "nullspace10.xml"
LOG_DIR = "runs/sac_free10dof"
MODELS_DIR = "models_sac_free10dof"
TOTAL_TIMESTEPS = 600_000
SEED = 12
MAX_EPISODE_STEPS = 500


# =================================================
#             PROGRESS CALLBACK
# =================================================
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=5):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.last_time = time.time()
        self.last_step = 0

    def _on_step(self) -> bool:
        now = time.time()
        if now - self.last_time >= self.print_freq:
            env = self.training_env.envs[0].unwrapped
            steps = self.num_timesteps

            qpos = env.data.qpos
            qvel = env.data.qvel

            base_ang = np.linalg.norm(qvel[3:6])
            joint_err = np.linalg.norm(qpos[7:])

            pct = 100 * steps / self.total_timesteps
            sps = (steps - self.last_step) / max(now - self.last_time, 1e-6)

            print(
                f"[{pct:6.2f}%] {steps:,}/{self.total_timesteps:,} | "
                f"SPS={sps:6.1f} | "
                f"base_ang={base_ang:.3f} | "
                f"joint_err={joint_err:.3f}"
            )

            self.last_time = now
            self.last_step = steps

        return True


# =================================================
#        RENDER EVERY N EPISODES CALLBACK
# =================================================
class PeriodicRenderCallback(BaseCallback):
    def __init__(self, render_every=100):
        super().__init__()
        self.render_every = render_every
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")

        if dones is not None and dones[0]:
            self.episode_count += 1

            if self.episode_count % self.render_every == 0:
                print(f"\n🎥 Rendering episode {self.episode_count}")

                env = self.training_env.envs[0].unwrapped
                env.render_enabled = True

                obs, _ = env.reset()
                done = False

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                env.render_enabled = False
                env.close()

        return True


# =================================================
#                  ENVIRONMENT
# =================================================
class HybridFree10DOFEnv(gym.Env):
    metadata = {}

    def __init__(self, xml_path=XML_PATH, seed=None, render=False):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.render_enabled = render
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.nq, self.nv, self.nu = self.model.nq, self.model.nv, self.model.nu

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.nu,),
            dtype=np.float32
        )

        # base quat (4) + base ang vel (3) + joints pos (4) + joints vel (4)
        obs_dim = 4 + 3 + 4 + 4

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.ctrl_max = np.abs(self.model.actuator_ctrlrange[:, 1])
        self.ctrl_max[self.ctrl_max == 0] = 1.0

        # -------- reward weights (ROTATION ONLY) --------
        self.w_joint = 2.0
        self.w_joint_vel = 0.1
        self.w_base_ang = 4.0
        self.w_action = 0.01

        self.success_bonus = 20.0
        self.step_count = 0

    def get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        return np.concatenate([
            qpos[3:7],           # base quaternion
            qvel[3:6] / 2.0,     # base angular velocity
            qpos[7:] / np.pi,    # joint positions
            qvel[6:] / 5.0       # joint velocities
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        self.data.qpos[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[7:] = np.random.uniform(-0.5, 0.5, size=4)

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

        return self.get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        self.data.ctrl[:] = action * self.ctrl_max
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        qpos = self.data.qpos
        qvel = self.data.qvel

        joint_err = np.linalg.norm(qpos[7:])
        joint_vel = np.linalg.norm(qvel[6:])
        base_ang = np.linalg.norm(qvel[3:6])

        reward = (
            - self.w_joint * joint_err
            - self.w_joint_vel * joint_vel
            - self.w_base_ang * base_ang
            - self.w_action * np.sum(action ** 2)
        )

        success = joint_err < 0.05 and base_ang < 0.1
        if success:
            reward += self.success_bonus

        reward = np.clip(reward, -50, 50)

        terminated = success
        truncated = self.step_count >= MAX_EPISODE_STEPS

        if self.render_enabled:
            self.render()

        return self.get_obs(), reward, terminated, truncated, {}

    def render(self):
        if not self.render_enabled:
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        time.sleep(1 / 60)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# =================================================
#                TRAINING
# =================================================
def make_env(seed):
    def _init():
        return Monitor(HybridFree10DOFEnv(XML_PATH, seed=seed, render=False))
    return _init


def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    env = DummyVecEnv([make_env(SEED)])
    eval_env = DummyVecEnv([make_env(SEED + 1)])

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=LOG_DIR,
        seed=SEED,
        verbose=1
    )

    callbacks = CallbackList([
        CheckpointCallback(50_000, MODELS_DIR, "sac_free10dof"),
        ProgressBarCallback(TOTAL_TIMESTEPS),
        PeriodicRenderCallback(render_every=100),
    ])

    model.learn(TOTAL_TIMESTEPS, callback=callbacks)
    model.save(os.path.join(MODELS_DIR, "sac_free10dof_final"))

    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Final Eval reward: {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    train()

