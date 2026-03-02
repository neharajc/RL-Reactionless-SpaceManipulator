import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

# ---------- Cubic trajectory ----------
def cubic_trajectory(q0, qf, T, steps=200):
    t = np.linspace(0, T, steps)
    q = q0 + (3*(qf - q0)/T**2) * t**2 - (2*(qf - q0)/T**3) * t**3
    dq = (6*(qf - q0)/T**2) * t - (6*(qf - q0)/T**3) * t**2
    return t, q, dq

# ---------- PID Controller ----------
class PIDController:
    def __init__(self, Kp, Kd, Ki=0.0):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.integral = 0.0

    def compute(self, q_des, q_actual, dq_des, dq_actual, dt):
        error = q_des - q_actual
        derror = dq_des - dq_actual
        self.integral += error * dt
        torque = self.Kp * error + self.Kd * derror + self.Ki * self.integral
        return torque

# ---------- Main ----------
def main():
    m = mujoco.MjModel.from_xml_path("xsimple.xml")
    d = mujoco.MjData(m)

    # Trajectory parameters
    T = 100000000000
    steps = 10
    goals = [0.5, 0.8, -0.5, 1.0, -1.0]
    starts = [0.0, 0.0, 0.0, 0.0, 0.0]

    q_trajs, dq_trajs = [], []
    for q0, qf in zip(starts, goals):
        t, q, dq = cubic_trajectory(q0, qf, T, steps)
        q_trajs.append(q)
        dq_trajs.append(dq)

    q_trajs = np.array(q_trajs)
    dq_trajs = np.array(dq_trajs)

    # Create PID controllers for each joint
    pids = [PIDController(Kp=1.0, Kd=1.0, Ki=1.0) for _ in range(5)]

    q_log = []
    dq_log = []
    tau_log = []

    dt = m.opt.timestep

    # Viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for i in range(steps):
            torque_cmd = np.zeros(m.nu)

            # Apply PID for each joint
            for j in range(5):
                torque_cmd[j] = pids[j].compute(
                    q_trajs[j][i], d.qpos[j], dq_trajs[j][i], d.qvel[j], dt
                )

            d.ctrl[:] = torque_cmd
            mujoco.mj_step(m, d)
            viewer.sync()

            q_log.append(d.qpos.copy())
            dq_log.append(d.qvel.copy())
            tau_log.append(d.actuator_force.copy())

            time.sleep(dt)

    # Convert to numpy
    q_log = np.array(q_log)
    dq_log = np.array(dq_log)
    tau_log = np.array(tau_log)

    # ---- Plot planned vs actual ----
    joint_names = ["base", "joint1", "joint2", "joint3", "joint4"]

    # Positions
    plt.figure(figsize=(12, 8))
    for j in range(5):
        plt.plot(t, q_trajs[j], 'r--', label=f"{joint_names[j]} planned" if j==0 else "")
        plt.plot(t, q_log[:, j], label=f"{joint_names[j]} actual")
    plt.xlabel("Time [s]"); plt.ylabel("Position [rad]")
    plt.legend(); plt.title("Planned vs Actual Joint Positions")
    plt.show()

    # Velocities
    plt.figure(figsize=(12, 8))
    for j in range(5):
        plt.plot(t, dq_trajs[j], 'r--', label=f"{joint_names[j]} planned" if j==0 else "")
        plt.plot(t, dq_log[:, j], label=f"{joint_names[j]} actual")
    plt.xlabel("Time [s]"); plt.ylabel("Velocity [rad/s]")
    plt.legend(); plt.title("Planned vs Actual Joint Velocities")
    plt.show()

    # Torques
    plt.figure(figsize=(12, 8))
    for j in range(m.nu):
        plt.plot(t, tau_log[:, j], label=f"Joint {j+1}")
    plt.xlabel("Time [s]"); plt.ylabel("Torque [Nm]")
    plt.legend(); plt.title("Applied Torques")
    plt.show()


if __name__ == "__main__":
    main()

