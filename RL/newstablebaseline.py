import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.evaluation import evaluate_policy

# ---------------- USER SETTINGS ----------------
XML_PATH = "nullspace.xml"   
LOG_DIR = "runs/sac_hybrid"
MODELS_DIR = "sac_models"
TOTAL_TIMESTEPS = 600_000
EVAL_EPISODES = 3
SEED = 0


class Hybrid5DOFEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path=XML_PATH, render=False, seed: int = None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found at: {xml_path}")

      
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

      
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.nu = int(self.model.nu)

     
        self.obs_dim = self.nq + self.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # ---- MuJoCo 3.x-safe name extraction ----
        # Actuator names
        self.actuator_names = []
        for i in range(self.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is None:
                name = f"actuator_{i}"
            self.actuator_names.append(name)

        # Joint names
       
        njnt = int(getattr(self.model, "njnt", max(0, self.nq)))  # fallback if njnt not present
        self.joint_names = []
        
        n_jnames = njnt if njnt > 0 else self.nq
        for i in range(n_jnames):
            jn = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jn is None:
                jn = f"joint_{i}"
            self.joint_names.append(jn)

       
        act_trnid = None
        try:
            act_trnid = np.array(self.model.actuator_trnid, dtype=int)
        except Exception:
            
            act_trnid = np.zeros((self.nu, 2), dtype=int) - 1

       
        self.actuator_to_joint = []
        for i in range(self.nu):
            tr_type, tr_id = int(act_trnid[i, 0]), int(act_trnid[i, 1])
            if tr_type == 0 and 0 <= tr_id < len(self.joint_names):
                self.actuator_to_joint.append(self.joint_names[tr_id])
            else:
               
                self.actuator_to_joint.append(self.actuator_names[i])

        # Identify base actuator index (if any) controlling 'base_joint'
        self.base_act_idx = None
        for i, jn in enumerate(self.actuator_to_joint):
            if jn == "base_joint":
                self.base_act_idx = i
                break

        
        self.exposed_act_idxs = [i for i in range(self.nu) if i != self.base_act_idx]

        
        if len(self.exposed_act_idxs) != 4:
            print("WARNING: Expected 4 exposed actuators, got", len(self.exposed_act_idxs))
            print("Actuator -> joint mapping:")
            for idx, (an, jn) in enumerate(zip(self.actuator_names, self.actuator_to_joint)):
                print(f"  idx {idx}: actuator '{an}' -> joint '{jn}'")
        else:
            print("Actuator mapping OK.")

       
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.exposed_act_idxs),), dtype=np.float32)

        # ctrlrange scaling (map agent [-1,1] to actuator ctrl space)
        try:
            ctrlrange = np.array(self.model.actuator_ctrlrange)
            if ctrlrange is None or ctrlrange.shape[0] != self.nu:
                raise Exception()
            self.ctrl_max = np.maximum(np.abs(ctrlrange[:, 0]), np.abs(ctrlrange[:, 1]))
            self.ctrl_max[self.ctrl_max == 0.0] = 1.0
        except Exception:
            self.ctrl_max = np.ones(self.nu, dtype=float)

        # Reward weights (joint-space reaching to zero for q[1:] -> joints 2..5)
        self.w_joint = 8.0     # penalty weight for actuated joints distance^2
        self.w_base = 5.0      # base qpos & qvel penalty
        self.w_action = 0.001  # small action magnitude penalty (agent-space)
        self.w_step = 0.002    # step penalty

        self.control_scaling = 1.0  # extra multiplier if desired
        self.render_enabled = render
        self.viewer = None

        # episode bookkeeping
        self.max_episode_steps = 500
        self.step_count = 0

        mujoco.mj_forward(self.model, self.data)

        # Print model summary for verification
        print("Loaded XML:", xml_path)
        print("nq, nv, nu:", self.nq, self.nv, self.nu)
        print("Actuator -> joint mapping:", list(zip(self.actuator_names, self.actuator_to_joint)))
        print("Exposed actuator indices (agent order):", self.exposed_act_idxs)
        if self.base_act_idx is not None:
            print("Masked base actuator index:", self.base_act_idx)

    def seed(self, seed=None):
        np.random.seed(seed)

    def get_obs(self):
        qpos = np.array(self.data.qpos[:self.nq], dtype=np.float32)
        qvel = np.array(self.data.qvel[:self.nv], dtype=np.float32)
        return np.concatenate([qpos, qvel])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # randomize initial joint angles from 0..10 degrees for all joints
        deg = np.random.uniform(0.0, 10.0, size=self.nq)
        self.data.qpos[:] = np.deg2rad(deg)
        # small random velocities
        self.data.qvel[:] = np.random.uniform(-0.05, 0.05, size=self.nv)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self.get_obs(), {}

    def compute_reward(self, agent_action):
        q = np.array(self.data.qpos[:self.nq])
        v = np.array(self.data.qvel[:self.nv])

        # actuated joints are q[1:] -> target 0
        joint_error = np.sum(q[1:] ** 2)

        # base penalty (q0, v0)
        base_pen = q[0] ** 2 + v[0] ** 2

        # action penalty (agent-space)
        action_pen = np.sum(np.array(agent_action) ** 2)

        r = - self.w_joint * joint_error - self.w_base * base_pen - self.w_action * action_pen - self.w_step
        return float(r)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape[0] == len(self.exposed_act_idxs), f"Action dim mismatch: got {action.shape} expected {len(self.exposed_act_idxs)}"

        # build full ctrl vector of size nu
        ctrl = np.zeros(self.nu, dtype=np.float32)

        # map agent actions to the corresponding actuator indices
        for i_agent, act_idx in enumerate(self.exposed_act_idxs):
            mapped = float(action[i_agent]) * float(self.ctrl_max[act_idx]) * self.control_scaling
            ctrl[act_idx] = mapped

        # ensure any non-exposed actuators (like base) are zeroed (force passive base)
        for i in range(self.nu):
            if i not in self.exposed_act_idxs:
                ctrl[i] = 0.0

        # apply ctrl and step simulation safely
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

        if self.render_enabled:
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print("Failed to launch mujoco.viewer:", e)
                self.viewer = None
                return
        try:
            self.viewer.sync()
        except Exception:
            pass

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

# ---------------- Callbacks from your reference (unchanged behavior) ----------------
class ConvergenceCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=50000,
                 reward_patience=15, min_eval_rewards=3,
                 warmup_steps=40000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.reward_patience = reward_patience
        self.min_eval_rewards = min_eval_rewards
        self.warmup_steps = warmup_steps
        self.eval_rewards = []
        self.best_reward = -np.inf
        self.no_improve_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self.warmup_steps:
            return True
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=3,
                deterministic=True,
                render=False
            )
            print(f"\n[Eval @ {self.num_timesteps}] reward = {mean_reward:.3f}")
            self.eval_rewards.append(mean_reward)
            if len(self.eval_rewards) < self.min_eval_rewards:
                return True
            if mean_reward > self.best_reward + 1e-3:
                self.best_reward = mean_reward
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                print(f"⚠ Plateau: {self.no_improve_count}/{self.reward_patience}")
            if self.no_improve_count >= self.reward_patience:
                print("\n EARLY STOP: Policy converged.")
                return False
        return True

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=5, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.last_print = time.time()
        self.start_time = time.time()
        self.last_step = 0

    def _on_step(self) -> bool:
        now = time.time()
        if now - self.last_print >= self.print_freq:
            elapsed = now - self.start_time
            steps = self.num_timesteps
            sps = (steps - self.last_step) / (now - self.last_print + 1e-8)
            pct = 100 * steps / max(1, self.total_timesteps)
            self.last_step = steps
            self.last_print = now
            print(f"[{pct:5.1f}%] Steps: {steps}/{self.total_timesteps} | SPS: {sps:7.1f} | Elapsed: {elapsed:7.1f}s")
        return True

# ---------------- Training function ----------------
def make_env_fn(xml_path=XML_PATH, seed=SEED):
    def _init():
        env = Hybrid5DOFEnv(xml_path=xml_path, render=False, seed=seed)
        return Monitor(env)
    return _init

def train(xml_path=XML_PATH, total_timesteps=TOTAL_TIMESTEPS):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    env = DummyVecEnv([make_env_fn(xml_path, SEED)])

    # eval env (non-vector)
    eval_env = Monitor(Hybrid5DOFEnv(xml_path=xml_path, render=False, seed=SEED+1))

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=LOG_DIR,
        seed=SEED,
    )

    checkpoint_cb = CheckpointCallback(save_freq=50_000, save_path=MODELS_DIR, name_prefix="sac_hybrid")
    convergence_cb = ConvergenceCallback(eval_env=eval_env, eval_freq=50_000)
    progress_cb = ProgressBarCallback(total_timesteps=total_timesteps)
    callbacks = CallbackList([checkpoint_cb, convergence_cb, progress_cb])

    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(os.path.join(MODELS_DIR, "sac_hybrid_final"))
    print("Training finished. Model saved to", MODELS_DIR)

    # quick evaluation run
    print("\nRunning quick evaluation (deterministic policy)...")
    model = SAC.load(os.path.join(MODELS_DIR, "sac_hybrid_final"))
    e = Hybrid5DOFEnv(xml_path=xml_path, render=False, seed=SEED+2)
    for ep in range(3):
        obs, _ = e.reset()
        done = False
        step = 0
        print(f"\n=== EVAL EPISODE {ep} ===")
        while not done and step < 400:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = e.step(action)
            qpos = obs[:e.nq]
            print(f"step {step:03d} | qpos_deg = {np.rad2deg(qpos).round(2)} | reward = {reward:.4f}")
            step += 1
            done = terminated or truncated
    e.close()

if __name__ == "__main__":
    train()

