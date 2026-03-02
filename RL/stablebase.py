import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback


# ================================================================
# 1. Custom MuJoCo Environment (SB3 Compatible)
# ================================================================
class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Load model
        self.model = mujoco.MjModel.from_xml_path("nullspace.xml")
        self.data = mujoco.MjData(self.model)

        # Joint IDs
        self.j1_id = self.model.joint("joint1").id
        self.j2_id = self.model.joint("joint2").id
        self.j3_id = self.model.joint("joint3").id
        self.j4_id = self.model.joint("joint4").id

        # Base DOF (unactuated)
        self.base_dof = self.model.joint("base_joint").dofadr[0]

        # ---- Observation space (10 values) ----
        # 5 qpos + 5 qvel
        high = np.ones(10) * np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # ---- Action space (4 torques) ----
        self.max_torque = 2.5
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=+self.max_torque,
            shape=(4,),
            dtype=np.float32
        )

        # Target angles (in degrees)
        self.target1 = 0.0
        self.target2 = 0.0

    # ------------------------------------------------------------------
    # Reset environment
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)

        # Random initial pose
        self.data.qpos[self.j1_id] = np.radians(np.random.uniform(0, 10))
        self.data.qpos[self.j2_id] = np.radians(np.random.uniform(0, 10))

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------
    def _get_obs(self):
        qpos = self.data.qpos[:5]
        qvel = self.data.qvel[:5]
        obs = np.concatenate([qpos, qvel])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Step environment
    # ------------------------------------------------------------------
    def step(self, action):

        # Enforce action bounds
        action = np.clip(action, -self.max_torque, self.max_torque)

        # Apply torques to 4 joints
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Observations
        obs = self._get_obs()

        # Joint angles (deg)
        j1 = np.degrees(self.data.qpos[self.j1_id])
        j2 = np.degrees(self.data.qpos[self.j2_id])

        # Compute error
        err = abs(j1 - self.target1) + abs(j2 - self.target2)

        # Base rotation penalty
        base_vel = abs(self.data.qvel[self.base_dof])

        # Reward
        reward = -err - 0.1 * base_vel - 0.01 * np.sum(action**2)

        # Termination condition
        done = err < 0.5

        if done:
            reward += 100.0

        return obs, reward, done, False, {}

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self):
        pass  # viewer handled separately


# ================================================================
# 2. Training Script for SAC
# ================================================================
def train_sac():

    env = RobotEnv()

    # Save checkpoints every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./sac_checkpoints/",
        name_prefix="robot_sac"
    )

    # SAC Model
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

    # Train for 1M steps
    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback
    )

    model.save("sac_robot_final")
    print("🎉 Training finished. Model saved!")


# ================================================================
# 3. Interactive Viewer for Trained Policy
# ================================================================
def run_viewer():

    env = RobotEnv()
    model = SAC.load("sac_robot_final")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, _ = env.reset()

        while viewer.is_running():

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            viewer.sync()


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    train_sac()
    # run_viewer()   # ← Uncomment after training
