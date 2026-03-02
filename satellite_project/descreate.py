import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------- Cubic trajectory ----------
def cubic_trajectory(q0, qf, T, steps=200):
    t = np.linspace(0, T, steps)
    q = q0 + (3*(qf - q0)/T**2) * t**2 - (2*(qf - q0)/T**3) * t**3
    dq = (6*(qf - q0)/T**2) * t - (6*(qf - q0)/T**3) * t**2
    return t, q, dq

# ---------- Discretize angle ----------
def discretize_angle(theta, bins=720):
    # map [-pi, pi] → [0, bins-1]
    idx = int((theta + np.pi) / (2*np.pi) * bins)
    return np.clip(idx, 0, bins-1)

# ---------- Main Simulation ----------
def main():
    model = mujoco.MjModel.from_xml_path("try.xml")
    data = mujoco.MjData(model)

    T = 10.0
    steps = 5000
    dt = model.opt.timestep

    # Desired trajectories for joint1 & joint2
    t, q1_traj, dq1_traj = cubic_trajectory(0.0, 1.5, T, steps)
    _, q2_traj, dq2_traj = cubic_trajectory(0.0, 1.5, T, steps)

    # Quality grid
    bins = 720  # 0.5 degree resolution
    quality_grid = np.zeros((bins, bins))
    visits = np.zeros((bins, bins))

    # -------- Viewer Loop --------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(steps):
            step_start = time.time()

            # step sim
            mujoco.mj_step(model, data)

            # actual vs desired
            q_des = np.array([q1_traj[i], q2_traj[i]])
            q_act = np.array([data.qpos[1], data.qpos[2]])

            # quality = negative squared tracking error
            quality = -np.sum((q_act - q_des)**2)

            # discretize angles
            idx1 = discretize_angle(data.qpos[1], bins)
            idx2 = discretize_angle(data.qpos[2], bins)

            # update grid
            quality_grid[idx1, idx2] += quality
            visits[idx1, idx2] += 1

            # sync viewer
            viewer.sync()

            # real-time pacing
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        print("Simulation finished. Close viewer window to exit.")

        # keep window open after sim
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

    # ---- Normalize and Plot Heatmap ----
    avg_quality = np.divide(
        quality_grid, visits,
        out=np.zeros_like(quality_grid),
        where=visits > 0
    )

    plt.figure(figsize=(8,6))
    plt.imshow(avg_quality, extent=[-180,180,-180,180], origin="lower")
    plt.colorbar(label="Quality (higher=better)")
    plt.xlabel("Joint1 angle (deg)")
    plt.ylabel("Joint2 angle (deg)")
    plt.title("State-space Quality Map")
    plt.show()

if __name__ == "__main__":
    main()
