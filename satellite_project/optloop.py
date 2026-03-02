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

# ---------- Main Simulation ----------
def main():
    # Load model
    model = mujoco.MjModel.from_xml_path("try.xml")
    data = mujoco.MjData(model)
        
    T = 10.0
    steps = 5000

    # Cubic trajectories for 2 commanded joints
    t, q1_traj, dq1_traj = cubic_trajectory(0.0, 1.5, T, steps)  # joint1
    _, q2_traj, dq2_traj = cubic_trajectory(0.0, 1.5, T, steps)  # joint2

    # Actuator IDs
    vel1_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel1")
    vel2_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel2")
    vel3_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel3")
    vel4_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel4")
    print("Actuator IDs:", vel1_id, vel2_id, vel3_id, vel4_id)

    # Logging
    q_log, dq_log, tau_log = [], [], []

    # Pause toggle
    paused = False
    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == ' ':
            paused = not paused

    # Viewer loop
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        i = 0
        dt = model.opt.timestep
        while viewer.is_running() and i < steps:
            step_start = time.time()

            # Desired velocities for commanded joints
            dq_des1 = dq1_traj[i]
            dq_des2 = dq2_traj[i]
            #dqdes=np.array([dq_des1,dq_des2])

            # Step physics
            if not paused:
                mujoco.mj_step(model, data)
            #print(f"print qdes: {data.qdes}")
            print(f"print qpos: {data.qpos}")

            # Mass matrix
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)

            # Constraint matrix (first row, base-joint coupling)
            Hbm = M[0:1, 1:5]   # shape (1,4)
            h1, h2 = Hbm[0,0], Hbm[0,1]

            # --- Solve Lagrange multiplier system numerically ---
            # 3x3 system: v1, v2, lambda
            A = np.array([
                [2, 0, h1],
                [0, 2, h2],
                [h1, h2, 0]
            ])
            b = np.array([2*dq_des1, 2*dq_des2, 0])

            # Solve for [v1_opt, v2_opt, lambda]
            try:
                x = np.linalg.solve(A, b)
                v1_opt, v2_opt = x[0], x[1]
            except np.linalg.LinAlgError:
                # fallback if singular
                v1_opt, v2_opt = dq_des1, dq_des2

            # Apply velocities to actuators
            data.ctrl[vel1_id] = v1_opt
            data.ctrl[vel2_id] = v2_opt
            # ⚠️ Leave vel3 and vel4 free

            # Logging
            q_log.append(data.qpos.copy())
            dq_log.append(data.qvel.copy())
            tau_log.append(data.actuator_force.copy())

            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            i += 1

    # Convert logs
    q_log = np.array(q_log)
    dq_log = np.array(dq_log)
    tau_log = np.array(tau_log)

    # --- Plot velocities for visualization ---
    plt.figure(figsize=(12, 6))
    plt.plot(t, dq1_traj, 'k--', label='Desired dq1')
    plt.plot(t, dq2_traj, 'r--', label='Desired dq2')
    plt.plot(dq_log[:, 0], label='Achieved dq0')
    plt.plot(dq_log[:, 1], label='Achieved dq1')
    plt.xlabel('Steps')
    plt.ylabel('Velocity [rad/s]')
    plt.legend()
    plt.title('Numerical Lagrange Multiplier Optimization (Joint1 & Joint2)')
    plt.show()


if __name__ == "__main__":
    main()

