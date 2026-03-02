import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

# ---------------- USER SETTINGS ----------------
XML_PATH = "nullspace.xml"
LOG_DIR = "runs/sac_hybrid1"
MODELS_DIR = "sac_models1"
TOTAL_TIMESTEPS = 600_000
SEED = 0

# ======================================================================= #
#                           ENVIRONMENT                                   #
# ======================================================================= #
class Hybrid5DOFEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path=XML_PATH, render=False, seed: int = None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.nq, self.nv, self.nu = int(self.model.nq), int(self.model.nv), int(self.model.nu)
        self.obs_dim = self.nq + self.nv
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # --- Joint and actuator mapping ---
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            for i in range(self.nq)
        ]
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
            for i in range(self.nu)
        ]

        try:
            act_trnid = np.array(self.model.actuator_trnid, dtype=int)
        except:
            act_trnid = np.zeros((self.nu, 2), dtype=int)

        self.actuator_to_joint = []
        for i in range(self.nu):
            tr_type, tr_id = act_trnid[i]
            if tr_type == 0 and 0 <= tr_id < len(self.joint_names):
                self.actuator_to_joint.append(self.joint_names[tr_id])
            else:
                self.actuator_to_joint.append(self.actuator_names[i])

        # --- Identify base actuator and exclude it from SAC ---
        self.base_act_idx = 0
        for i, jn in enumerate(self.actuator_to_joint):
            if jn == "base_joint":
                self.base_act_idx = i
                break

        self.exposed_act_idxs = [i for i in range(self.nu) if i != self.base_act_idx]
        self.action_space = spaces.Box(-1, 1, shape=(len(self.exposed_act_idxs),), dtype=np.float32)

        # --- ctrlrange scaling ---
        try:
            cr = np.array(self.model.actuator_ctrlrange)
            self.ctrl_max = np.max(np.abs(cr), axis=1)
            self.ctrl_max[self.ctrl_max == 0] = 1.0
        except:
            self.ctrl_max = np.ones(self.nu)

        # --- Reward weights ---
        self.w_joint = 8.0
        self.w_base = 5.0
        self.w_action = 0.001
        self.w_step = 0.002

        self.max_episode_steps = 500
        self.step_count = 0

        # Torque smoothing
        self.prev_ctrl = np.zeros(self.nu, dtype=np.float32)
        self.max_torque_change = 0.05

        mujoco.mj_forward(self.model, self.data)

        print("Loaded XML:", xml_path)
        print("Exposed actuators:", self.exposed_act_idxs)
        print("Base actuator index:", self.base_act_idx)

    def get_obs(self):
        return np.concatenate([
            np.array(self.data.qpos[:self.nq], dtype=np.float32),
            np.array(self.data.qvel[:self.nv], dtype=np.float32)
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.base_act_idx is not None:
            self.data.qpos[self.base_act_idx] = 0.0
            self.data.qvel[self.base_act_idx] = 0.0

        for j in range(1, self.nq):
            self.data.qpos[j] = np.deg2rad(np.random.uniform(0, 10))
            self.data.qvel[j] = np.random.uniform(-0.05, 0.05)

        mujoco.mj_forward(self.model, self.data)
        self.prev_joint_error = np.sum(self.data.qpos[1:] ** 2)
        self.step_count = 0
        return self.get_obs(), {}

    def compute_reward(self, agent_action):
        q = np.array(self.data.qpos)
        v = np.array(self.data.qvel)

        joint_error = np.sum(q[1:] ** 2)

        angle_limit = np.deg2rad(10.0)
        limit_penalty = np.sum(np.maximum(0, np.abs(q[1:]) - angle_limit) ** 2)

        base_pen = q[0] ** 2 + v[0] ** 2
        action_pen = np.sum(agent_action ** 2)
        step_pen = self.w_step

        reward = (
            - self.w_joint * joint_error
            - 10.0 * limit_penalty
            - self.w_base * base_pen
            - self.w_action * action_pen
            - step_pen
        )

        improvement = self.prev_joint_error - joint_error
        reward += 2.0 * improvement
        self.prev_joint_error = joint_error

        if abs(v[0]) < 0.05:
            reward += 0.5

        if np.linalg.norm(agent_action) < 0.3:
            reward += 0.2

        if joint_error < 0.01:
            reward += 2.0
            print("✅ target reached")

        return float(reward)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # Raw torque
        raw_ctrl = np.zeros(self.nu, dtype=np.float32)
        for i_agent, act_idx in enumerate(self.exposed_act_idxs):
            raw_ctrl[act_idx] = action[i_agent] * self.ctrl_max[act_idx]

        raw_ctrl[self.base_act_idx] = 0.0

        # Smoothing
        ctrl = self.prev_ctrl + np.clip(
            raw_ctrl - self.prev_ctrl,
            -self.max_torque_change,
            self.max_torque_change
        )
        self.prev_ctrl = ctrl.copy()

        self.data.ctrl[:] = ctrl

        mujoco.mj_step1(self.model, self.data)
        mujoco.mj_step2(self.model, self.data)

        self.step_count += 1
        obs = self.get_obs()
        reward = self.compute_reward(action)

        terminated = False
        truncated = False
        info = {}

        if self.step_count >= self.max_episode_steps:
            truncated = True
            info["TimeLimit.truncated"] = True

        if self.step_count % 10 == 0:
            print(f"Step {self.step_count:03d} | qpos: {self.data.qpos[:self.nq]}")

        return obs, reward, terminated, truncated, info


# ======================================================================= #
#                           CALLBACKS                                     #
# ======================================================================= #
class ConvergenceCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=50000, reward_patience=8, min_eval_rewards=3, warmup_steps=40000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.reward_patience = reward_patience
        self.min_eval_rewards = min_eval_rewards
        self.warmup_steps = warmup_steps
        self.eval_rewards = []
        self.best_reward = -np.inf
        self.no_improve_count = 0

    def _on_step(self):
        if self.num_timesteps < self.warmup_steps:
            return True

        if self.num_timesteps % self.eval_freq == 0:

            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=3,
                deterministic=True
            )

            print(f"[Eval {self.num_timesteps}] reward={mean_reward:.3f}")

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            if self.no_improve_count >= self.reward_patience:
                print("\nEARLY STOP: converged.")
                return False

        return True


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=5):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.last_t = time.time()
        self.last_step = 0

    def _on_step(self):
        now = time.time()
        if now - self.last_t >= self.print_freq:
            steps = self.num_timesteps
            sps = (steps - self.last_step) / (now - self.last_t)
            pct = 100 * steps / self.total_timesteps
            print(f"[{pct:5.1f}%] {steps}/{self.total_timesteps} | SPS={sps:.1f}")
            self.last_t = now
            self.last_step = steps
        return True


# ======================================================================= #
#                           TRAINING                                      #
# ======================================================================= #
def make_env_fn(xml_path=XML_PATH, seed=SEED):
    def _init():
        return Monitor(Hybrid5DOFEnv(xml_path=xml_path, render=False, seed=seed))
    return _init


def train(xml_path=XML_PATH, total_timesteps=TOTAL_TIMESTEPS):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    env = DummyVecEnv([make_env_fn(xml_path, SEED)])
    eval_env = Monitor(Hybrid5DOFEnv(xml_path, seed=SEED + 1))

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=LOG_DIR,
        seed=SEED
    )

    callbacks = CallbackList([
        CheckpointCallback(save_freq=50000, save_path=MODELS_DIR, name_prefix="sac_hybrid"),
        ConvergenceCallback(eval_env),
        ProgressBarCallback(total_timesteps)
    ])

    print("\nStarting SAC training...\n")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(os.path.join(MODELS_DIR, "sac_hybrid_final1"))
    print("\nTraining complete.")

    print("\nRunning quick evaluation...")
    e = Hybrid5DOFEnv(xml_path, seed=SEED + 2)
    for ep in range(3):
        obs, _ = e.reset()
        for step in range(400):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = e.step(action)
            print(f"step {step:03d} | reward={reward:.3f}")
            if terminated or truncated:
                break
    e.close()


if __name__ == "__main__":
    train()
