# stablebase.py
"""
Stable-Baselines3 SAC training for your MuJoCo hybrid_robot model.
Fix: environment expects 5 MuJoCo actuators but SAC policy outputs 4 actions
(we do not control base actuator 0). The env pads the policy action into a
5-length control vector so `data.ctrl[:]` receives shape (5,).
"""

import os
import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# -------------------------
# Environment
# -------------------------
class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, max_steps=750):
        super().__init__()

        # Load MuJoCo model (XML file must be in working directory)
        self.model = mujoco.MjModel.from_xml_path("nullspace.xml")  # or nullspace.xml if that is your file
        self.data = mujoco.MjData(self.model)

        # Joint indices (as in your XML)
        self.j1_id = self.model.joint("joint1").id
        self.j2_id = self.model.joint("joint2").id
        self.j3_id = self.model.joint("joint3").id
        self.j4_id = self.model.joint("joint4").id

        # Base DOF index (we don't control it)
        self.base_dof = self.model.joint("base_joint").dofadr[0]

        # Number of actuators expected by MuJoCo
        self.nu = self.model.nu  # should be 5 for your XML

        # Action space: policy outputs 4 values (for joint1..joint4)
        # They will be mapped into actuators idx 1..4; actuator 0 (base) kept zero
        self.max_torque = 2.5
        self.action_space = spaces.Box(low=-self.max_torque,
                                       high=self.max_torque,
                                       shape=(4,),
                                       dtype=np.float32)

        # Observation space: qpos (5) + qvel (5) => 10-dim vector
        obs_high = np.full((10,), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Targets and training params
        self.target1 = 0.0  # degrees
        self.target2 = 0.0  # degrees
        self.max_steps = max_steps
        self.step_count = 0

        # Initialize sim
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        # get the first 5 qpos and 5 qvel (as in earlier design)
        qpos = np.array(self.data.qpos[:5], dtype=np.float32)
        qvel = np.array(self.data.qvel[:5], dtype=np.float32)
        obs = np.concatenate([qpos, qvel])
        return obs

    def reset(self, *, seed=None, options=None):
        # Reset sim data and optionally randomize initial joints
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # randomize first two joints as in original script
        self.data.qpos[self.j1_id] = np.radians(np.random.uniform(0, 10))
        self.data.qpos[self.j2_id] = np.radians(np.random.uniform(0, 10))

        # ensure forward dynamics & positions consistent
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        """
        action: array of shape (4,) from the policy, representing torques/vel commands
                for joint1..joint4. We must supply shape (nu,) to data.ctrl. We set
                base actuator (index 0) to 0.0 and place action into indices 1..4.
        """

        # sanity: clip action
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_torque, self.max_torque)

        # Build full control vector of length self.nu (5)
        # IMPORTANT: keep base actuator at index 0 as zero (we don't control base)
        full_ctrl = np.zeros(self.nu, dtype=np.float32)

        # Map policy action to actuators 1..4 (matches your XML ordering)
        # If your actuators ordering is different, adjust these indices accordingly.
        if self.nu >= 5:
            full_ctrl[1:5] = action[:4]
        else:
            # safety fallback (unlikely for your XML)
            full_ctrl[:action.shape[0]] = action

        # Apply controls and step
        self.data.ctrl[:] = full_ctrl
        mujoco.mj_step(self.model, self.data)

        # Observations
        obs = self._get_obs()

        # Compute reward (same spirit as your Q-learning reward)
        j1_deg = float(np.degrees(self.data.qpos[self.j1_id]))
        j2_deg = float(np.degrees(self.data.q2_id if False else self.data.qpos[self.j2_id]))  # safe read
        # Note: above line uses safe read; j2_deg set from qpos index.

        # simpler: compute error for first two joints (in degrees)
        j1 = np.degrees(self.data.qpos[self.j1_id])
        j2 = np.degrees(self.data.qpos[self.j2_id])
        err = abs(j1 - self.target1) + abs(j2 - self.target2)

        # base velocity penalty (stabilization)
        base_vel = float(self.data.qvel[self.base_dof])
        base_penalty = 0.1 * abs(base_vel)

        # control penalty (sum squared on policy action)
        control_penalty = 0.01 * float(np.sum(action**2))

        reward = -err - base_penalty - control_penalty

        done = False
        self.step_count += 1

        # Terminal bonus if near target (same threshold as original script)
        if err < 0.5:
            reward += 100.0
            done = True

        # Truncate if too long
        truncated = self.step_count >= self.max_steps

        info = {}

        return obs, float(reward), bool(done), bool(truncated), info

    # Optional: separate visual viewer function
    def render(self, mode="human"):
        # viewer will be handled externally using mujoco.viewer.launch_passive
        pass

# -------------------------
# Training (Stable-Baselines3)
# -------------------------
def train_sac(total_timesteps=200_000):
    env = RobotEnv(max_steps=750)

    # Checkpoint callback: saves every 50k steps
    checkpoint_callback = CheckpointCallback(save_freq=50_000,
                                             save_path="./sac_checkpoints/",
                                             name_prefix="robot_sac")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("sac_robot_final")
    print("Training finished and model saved to sac_robot_final.zip")

# -------------------------
# Optional: run trained policy in viewer
# -------------------------
def run_viewer(model_path="sac_robot_final"):
    env = RobotEnv(max_steps=750)
    model = SAC.load(model_path)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, _ = env.reset()
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            viewer.sync()
            if done or truncated:
                obs, _ = env.reset()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Train (change timesteps as you like)
    train_sac(total_timesteps=200_000)
    # After training, run the viewer:
    # run_viewer("sac_robot_final")

