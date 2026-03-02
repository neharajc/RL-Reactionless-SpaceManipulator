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
XML_PATH = "nullspace.xml"
LOG_DIR = "runs/sac_fixed_ideal"
MODELS_DIR = "models_sac_fixed"
TOTAL_TIMESTEPS = 600_000
SEED = 12
MAX_EPISODE_STEPS = 300

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
            steps = self.num_timesteps
            env = self.training_env.envs[0].unwrapped  # <-- fixed indexing

            qpos = env.data.qpos.copy()
            joint_error = np.sum(np.abs(qpos[1:5]))  # joints 1-4
            joint_l2 = np.sum(qpos[1:5] ** 2)
            base_vel = env.data.qvel[0]  # use first DOF for base

            pct = 100.0 * steps / self.total_timesteps
            sps = (steps - self.last_step) / max(now - self.last_time, 1e-6)

            print(f"[{pct:6.2f}%] {steps:,}/{self.total_timesteps:,} | "
                  f"SPS={sps:6.1f} | base_vel={base_vel:+.4f} | "
                  f"joint_err={joint_error:.4f} | w_base_vel={env.w_base_vel:.2f}")

            self.last_time = now
            self.last_step = steps
        return True

# =================================================
#                  ENVIRONMENT (FIXED)
# =================================================
class Hybrid5DOFEnv(gym.Env):
    def __init__(self, xml_path=XML_PATH, seed=None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.nq, self.nv, self.nu = self.model.nq, self.model.nv, self.model.nu
        self.exposed_act_idxs = list(range(1, self.nu))  # joints 1+

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self.exposed_act_idxs),),
            dtype=np.float32
        )

        obs_dim = (self.nq - 1) + (self.nv - 1) + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.ctrl_max = np.abs(self.model.actuator_ctrlrange[:, 1])
        self.ctrl_max[self.ctrl_max == 0] = 1.0

        # Reward weights
        self.w_progress = 8.0
        self.w_joint = 5.0
        self.w_base_vel = 2.0
        self.w_base_acc = 0.5
        self.w_action = 0.01
        self.w_reaction = 0.1
        self.success_bonus = 50.0

        self.prev_joint_error = 0.0
        self.prev_base_vel = 0.0
        self.step_count = 0

    def get_obs(self):
        q = self.data.qpos
        v = self.data.qvel
        obs = np.concatenate([
            q[1:] / np.deg2rad(20),
            v[1:] / 2.0,
            np.array([np.clip(v[0] / 3.0, -1.0, 1.0)])
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        # Random initial joints 1+
        for j in range(1, self.nq):
            self.data.qpos[j] = np.random.uniform(-0.8, 0.8)

        mujoco.mj_forward(self.model, self.data)

        self.prev_joint_error = np.sum(np.abs(self.data.qpos[1:5]))
        self.prev_base_vel = 0.0
        self.step_count = 0
        return self.get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ctrl = np.zeros(self.nu)
        for i, idx in enumerate(self.exposed_act_idxs):
            ctrl[idx] = action[i] * self.ctrl_max[idx]

        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        q = self.data.qpos
        v = self.data.qvel

        joint_error = np.sum(np.abs(q[1:5]))
        joint_l2 = np.sum(q[1:5] ** 2)

        base_vel = np.clip(v[0], -5.0, 5.0)
        base_acc = np.clip((base_vel - self.prev_base_vel) / self.model.opt.timestep, -20.0, 20.0)
        self.prev_base_vel = base_vel

        reaction_cost = abs(base_vel) * np.sum(np.abs(v[1:]))
        improvement = self.prev_joint_error - joint_error
        self.prev_joint_error = joint_error

        reward = (
            self.w_progress * improvement
            - self.w_joint * joint_error
            - 10.0 * joint_l2
            - self.w_base_vel * abs(base_vel)
            - self.w_base_acc * abs(base_acc)
            - self.w_reaction * reaction_cost
            - self.w_action * np.sum(action ** 2)
        )

        is_success = joint_error < 0.03 and abs(base_vel) < 0.03
        if is_success:
            reward += self.success_bonus

        reward = np.clip(reward, -50.0, 50.0)
        terminated = False
        truncated = self.step_count >= MAX_EPISODE_STEPS

        info = {
            "joint_error": joint_error,
            "joint_l2": joint_l2,
            "base_velocity": base_vel,
            "is_success": is_success,
            "reward": reward
        }

        return self.get_obs(), float(reward), terminated, truncated, info

# =================================================
#                CURRICULUM CALLBACK
# =================================================
class CurriculumCallback(BaseCallback):
    def _on_step(self) -> bool:
        env = self.training_env.envs[0].unwrapped
        steps = self.num_timesteps

        if steps < 200_000:
            env.w_joint = 4.0
            env.w_base_vel = 1.0
            env.w_progress = 6.0
        elif steps < 400_000:
            env.w_joint = 6.0
            env.w_base_vel = 2.5
            env.w_progress = 8.0
        else:
            env.w_joint = 8.0
            env.w_base_vel = 3.0
            env.w_progress = 10.0

        return True

# =================================================
#                TRAINING
# =================================================
def make_env(seed):
    def _init():
        return Monitor(Hybrid5DOFEnv(XML_PATH, seed=seed))
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
        CheckpointCallback(save_freq=50_000, save_path=MODELS_DIR, name_prefix="sac_fixed"),
        ProgressBarCallback(TOTAL_TIMESTEPS),
        CurriculumCallback()
    ])

    model.learn(TOTAL_TIMESTEPS, callback=callbacks)
    model.save(os.path.join(MODELS_DIR, "sac_ideal_final_fixed"))

    print("✅ Training finished!")
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Final Eval reward: {mean:.2f} ± {std:.2f}")

if __name__ == "__main__":
    train()

