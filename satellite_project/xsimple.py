import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

def cubic_trajectory(q0, qf, T, steps=200):
    t = np.linspace(0, T, steps)
    q = q0 + (3*(qf - q0)/T**2) * t**2 - (2*(qf - q0)/T**3) * t**3
    dq = (6*(qf - q0)/T**2) * t - (6*(qf - q0)/T**3) * t**2
    ddq = (6*(qf - q0)/T**2) - (12*(qf - q0)/T**3) * t
    return t, q, dq, ddq

def main():
    m = mujoco.MjModel.from_xml_path("xsimple.xml")
    d = mujoco.MjData(m)

    T = 5.0
    steps = 3000

    q_trajs, dq_trajs, ddq_trajs = [], [], []
    goals = [0.5, 0.8, -0.5, 1.0, -1.0]
    starts = [0.0, 0.0, 0.0, 0.0, 0.0]

    for q0, qf in zip(starts, goals):
        t, q, dq, ddq = cubic_trajectory(q0, qf, T, steps)
        q_trajs.append(q)
        dq_trajs.append(dq)
        ddq_trajs.append(ddq)

    q_trajs = np.array(q_trajs)
    dq_trajs = np.array(dq_trajs)
    ddq_trajs = np.array(ddq_trajs)

    # Launch viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.0, 0.0, 0.2]
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -30

        for i in range(1000):
            torque_cmd = np.zeros(m.nu)

            # Example: applying torque to joint3 & joint4
            torque_cmd[3] = 5.0
            torque_cmd[4] = 1.0

            d.ctrl[:] = torque_cmd
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.001)
            print(d.qpos)

    # ---- Plot cubic trajectories ----
    joint_names = ["base", "joint1", "joint2", "joint3", "joint4"]

    plt.figure(figsize=(10, 8))

    # Positions
    plt.subplot(3, 1, 1)
    for i in range(5):
        plt.plot(t, q_trajs[i], label=joint_names[i])
    plt.ylabel("Position [rad]")
    plt.legend()

    # Velocities
    plt.subplot(3, 1, 2)
    for i in range(5):
        plt.plot(t, dq_trajs[i], label=joint_names[i])
    plt.ylabel("Velocity [rad/s]")

    # Accelerations
    plt.subplot(3, 1, 3)
    for i in range(5):
        plt.plot(t, ddq_trajs[i], label=joint_names[i])
    plt.ylabel("Acceleration [rad/s²]")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

