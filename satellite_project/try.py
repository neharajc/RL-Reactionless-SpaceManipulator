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

    # --- Joint mapping ---
    print("\n--- Joint mapping (qpos order) ---")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"qpos[{i}] → joint '{name}'")

    # --- Actuator mapping ---
    print("\n--- Actuator mapping (ctrl order) ---")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model.actuator_trnid[i, 0]  # joint id
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
        print(f"ctrl[{i}] actuator '{name}' → joint '{jname}'")

    # Trajectory
    steps = 5000
    T = 10.0
    q0, qf = 0.0, 1.5
    q1, qf1 = 0.0, 1.5
    q2, qf2 = 0.0, 1.5
    q3, qf3 = 0.0, 1.5
    
    t, q_traj, dq_traj = cubic_trajectory(q0, qf, T, steps)
    t1, q1_traj,dq1_traj = cubic_trajectory(q1, qf1, T, steps)
    t2, q2_traj,dq2_traj = cubic_trajectory(q2, qf2, T, steps)
    t3, q3_traj,dq3_traj = cubic_trajectory(q3, qf3, T, steps)

    # Actuator IDs
    vel1_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel1")
    vel2_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel2")
    vel3_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel3")
    vel4_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel4")
    motor0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor0")
    print("Actuator IDs:", vel1_id, vel2_id, vel3_id, vel4_id, motor0_id)

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

            # Desired state
            q_des, dq_des = q_traj[i], dq_traj[i]
            q1_des, dq1_des = q1_traj[i], dq1_traj[i]
            q2_des, dq2_des = q2_traj[i], dq2_traj[i]
            q3_des, dq3_des = q3_traj[i], dq3_traj[i]

            # Apply controls
            data.ctrl[motor0_id] = 0.0
            data.ctrl[vel1_id] = dq_des
            data.ctrl[vel2_id] = dq1_des
            data.ctrl[vel3_id] = dq2_des
            data.ctrl[vel4_id] = dq3_des

            if not paused:
                mujoco.mj_step(model, data)

            # Print joint positions
            print(data.qpos)

            # ---- Mass matrix of the system ----
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)
            print(f"Mass matrix at step {i}:\n{M}\n")

            # Log
            q_log.append(data.qpos.copy())
            dq_log.append(data.qvel.copy())
            tau_log.append(data.actuator_force.copy())

            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            i += 1

    # Post-run viewer hold
    end_time = time.time() + 5.0
    while viewer.is_running() and time.time() < end_time:
        viewer.sync()
        time.sleep(0.01)

    # Convert logs
    q_log = np.array(q_log)
    dq_log = np.array(dq_log)
    tau_log = np.array(tau_log)

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    for j in range(model.nq):
        plt.plot(t, q_log[:, j], label=f'Joint {j}')
    plt.plot(t, q_traj, 'k--', label='Desired Joint1')
    plt.plot(t, q1_traj, 'r--', label='Desired Joint2')
    plt.xlabel('Time [s]'); plt.ylabel('Position [rad]')
    plt.legend(); plt.title('Joint Positions'); plt.show()

    plt.figure(figsize=(12, 8))
    for j in range(model.nv):
        plt.plot(t, dq_log[:, j], label=f'Joint {j}')
    plt.xlabel('Time [s]'); plt.ylabel('Velocity [rad/s]')
    plt.legend(); plt.title('Joint Velocities'); plt.show()

    plt.figure(figsize=(12, 8))
    for a in range(model.nu):
        plt.plot(t, tau_log[:, a], label=f'Actuator {a}')
    plt.xlabel('Time [s]'); plt.ylabel('Torque [Nm]')
    plt.legend(); plt.title('Actuator Torques'); plt.show()


if __name__ == "__main__":
    main()

