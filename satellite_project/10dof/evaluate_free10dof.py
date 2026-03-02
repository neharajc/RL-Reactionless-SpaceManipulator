import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from sac_code1 import HybridFree10DOFEnv, XML_PATH, MAX_EPISODE_STEPS


MODEL_PATH = "models_sac_free10dof/sac_free10dof_final.zip"
SEED = 123


# =====================================================
#               ENV CREATION
# =====================================================
def make_env(render=False):
    def _init():
        return Monitor(HybridFree10DOFEnv(
            XML_PATH,
            seed=SEED,
            render=render
        ))
    return _init


# =====================================================
#            EVALUATION + LOGGING
# =====================================================
def evaluate_and_plot(render=True):
    env = DummyVecEnv([make_env(render=render)])
    model = SAC.load(MODEL_PATH, env=env)

    obs = env.reset()

    base_ori_err = []
    base_ang_vel = []

    joint_err = []   # shape: [T, 4]
    reward_log = []

    for step in range(MAX_EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if dones[0]:
            break

        unwrapped = env.envs[0].unwrapped
        qpos = unwrapped.data.qpos.copy()
        qvel = unwrapped.data.qvel.copy()

        # ---------------- BASE ----------------
        qw = np.clip(abs(qpos[3]), -1.0, 1.0)
        base_theta = 2 * np.arccos(qw)     # orientation error (rad)
        base_w = np.linalg.norm(qvel[3:6]) # angular velocity

        # ---------------- JOINTS ----------------
        joints = qpos[7:11]  # 4 joints

        base_ori_err.append(base_theta)
        base_ang_vel.append(base_w)
        joint_err.append(joints)
        reward_log.append(rewards[0])

        if dones[0]:
            print(f"Episode finished at step {step}")
            break

    env.close()

    base_ori_err = np.array(base_ori_err)
    base_ang_vel = np.array(base_ang_vel)
    joint_err = np.array(joint_err)
    reward_log = np.array(reward_log)

    plot_results(base_ori_err, base_ang_vel, joint_err, reward_log)


# =====================================================
#                    PLOTS
# =====================================================
def plot_results(base_ori, base_w, joint_err, rewards):
    t = np.arange(len(rewards))

    # ---- Base Orientation Error ----
    plt.figure()
    plt.plot(t, base_ori, linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Base orientation error (rad)")
    plt.title("Base Orientation Error")
    plt.grid()
    plt.savefig("eval_base_orientation.png")
    plt.close()

    # ---- Base Angular Velocity ----
    plt.figure()
    plt.plot(t, base_w, linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("||Base angular velocity||")
    plt.title("Base Angular Velocity")
    plt.grid()
    plt.savefig("eval_base_ang_vel.png")
    plt.close()

    # ---- Joint Errors (INDIVIDUAL) ----
    plt.figure()
    for i in range(joint_err.shape[1]):
        plt.plot(t, joint_err[:, i], label=f"Joint {i+1}")
    plt.xlabel("Timestep")
    plt.ylabel("Joint angle (rad)")
    plt.title("Joint Errors")
    plt.legend()
    plt.grid()
    plt.savefig("eval_joint_errors.png")
    plt.close()

    # ---- Reward ----
    plt.figure()
    plt.plot(t, rewards, linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward per step")
    plt.grid()
    plt.savefig("eval_reward.png")
    plt.close()

    print("\n✅ Evaluation complete")
    print("Saved plots:")
    print("  eval_base_orientation.png")
    print("  eval_base_ang_vel.png")
    print("  eval_joint_errors.png")
    print("  eval_reward.png")


# =====================================================
#                    MAIN
# =====================================================
if __name__ == "__main__":
    evaluate_and_plot(render=True)

