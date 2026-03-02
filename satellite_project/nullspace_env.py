import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time

class NullspaceEnv(gym.Env):
    def __init__(self, model_path="nullspace.xml", steps=500, render_mode=False):
        super(NullspaceEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # --- Smaller timestep for smoother motion ---
        self.model.opt.timestep = 0.001  # 1 ms per physics step

        self.steps = steps
        self.current_step = 0
        self.render_mode = render_mode
        self.viewer = None  # viewer handle

        nv = self.model.nv
        self.base_dof = 0
        self.arm_dofs = list(range(1, nv))

        # Desired arm positions (targets)
        self.arm_target = np.array([0.5, 1.0, 0.5, 1.0], dtype=np.float32)

        # --- Action / Observation spaces ---
        self.action_space = spaces.Box(low=-8, high=8, shape=(4,), dtype=np.float32)
        n_obs = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        print("Number of actuators (model.nu):", self.model.nu)
        print("Shape of data.ctrl:", self.data.ctrl.shape)

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        if self.render_mode and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        # --- Apply actions to arm joints ---
        self.data.ctrl[:] = 0.0
        self.data.ctrl[1:5] = action  # control only arm joints
        mujoco.mj_step(self.model, self.data)

        # --- Viewer sync ---
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(0.001)

        # --- Rewards ---
        base_pos = float(self.data.qpos[self.base_dof])
        base_vel = float(self.data.qvel[self.base_dof])

        # Strong penalty for base motion
        reward_base = -10.0 * (abs(base_pos) + abs(base_vel))

        # Arm tracking reward
        arm_pos = self.data.qpos[1:5]
        err = self.arm_target - arm_pos
        reward_arm = -np.linalg.norm(err)

        # Bonus for reaching near target
        if np.all(np.abs(err) < 0.05):
            reward_arm += 10.0

        reward = reward_base + reward_arm

        # --- Logging ---
        print(f"Step {self.current_step}: Base Pos: {base_pos:.4f}, Base Ori: {self.data.qpos[1:4]}, "
              f"Arm Pos: {arm_pos}, Reward: {reward:.3f}")

        self.current_step += 1
        done = self.current_step >= self.steps

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception as e:
                print("Viewer close error (ignored):", e)
            self.viewer = None

