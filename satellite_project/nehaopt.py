import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# ---------- Cubic trajectory ----------
def cubic_trajectory(q0, qf, T, steps=200):
    t = np.linspace(0, T, steps)
    q = q0 + (3*(qf - q0)/T**2) * t**2 - (2*(qf - q0)/T**3) * t**3
    dq = (6*(qf - q0)/T**2) * t - (6*(qf - q0)/T**3) * t**2
    return t, q, dq

# ---------- Null-space projection via QP ----------
def nullspace_project(Hbm, v_des, reg=1e-12):
    """
    Solve min ||v - v_des||^2 s.t. Hbm v = 0
    """
    H = Hbm.astype(float)
    m, n = H.shape
    if m == 0:
        return v_des.copy()
   
    # Lagrange multiplier solution
    S = H @ H.T
    S_reg = S + reg * np.eye(m)
    lam = np.linalg.solve(S_reg, (H @ v_des).reshape(-1,))
    v_cmd = v_des - (H.T @ lam).reshape(-1,)
    return v_cmd

# ---------- Main Simulation ----------
def main():
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path("try.xml")
    data = mujoco.MjData(model)

    # Simulation parameters
    T = 10.0
    steps = 5000
    dt = model.opt.timestep

    # Trajectories for 4 manipulator joints
    trajs = [cubic_trajectory(0.0, 1.5, T, steps) for _ in range(4)]
    q_traj_all = [tr[1] for tr in trajs]
    dq_traj_all = [tr[2] for tr in trajs]

    # Actuator IDs
    vel_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vel{i+1}") for i in range(4)]
    motor0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor0")

    # Gains / limits
    Kp = 50.0
    Kd = 5.0
    kp_base = 6.0
    kd_base = 1.0
    max_joint_vel = 8.0
    reg_proj = 1e-10

    # Indices
    nv = model.nv
    base_dof = 0
    arm_dofs = list(range(1, nv))  # manipulator DOFs

    # Logging
    q_log, dq_log, vdes_log, vcmd_log, hres_log = [], [], [], [], []

    paused = False
    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == ' ':
            paused = not paused

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        Mfull = np.zeros((nv, nv))
        for i in range(steps):
            step_start = time.time()

            # Current state
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()

            # Desired velocities (PD on joint positions)
            q_des = np.array([q_traj_all[j][i] for j in range(4)])
            dq_des = np.array([dq_traj_all[j][i] for j in range(4)])
            v_des = dq_des + Kp * (q_des - qpos[1:5]) + Kd * (dq_des - qvel[1:5])
            v_des = np.clip(v_des, -max_joint_vel, max_joint_vel)

            # Compute full mass matrix
            mujoco.mj_fullM(model, Mfull, data.qM)
            Hbm = Mfull[np.ix_([base_dof], arm_dofs)]

            # Quadratic optimization / null-space projection
            v_cmd = nullspace_project(Hbm, v_des, reg=reg_proj)
            v_cmd = np.clip(v_cmd, -max_joint_vel, max_joint_vel)

            # Base PD controller (numerical safeguard)
            base_pos = float(qpos[base_dof])
            base_vel = float(qvel[base_dof])
            data.ctrl[motor0_id] = float(-kp_base * base_pos - kd_base * base_vel)

            # Send velocity commands to actuators
            for j, aid in enumerate(vel_ids):
                data.ctrl[aid] = float(v_cmd[j])

            # Step physics
            if not paused:
                mujoco.mj_step(model, data)

            # Logging
            q_log.append(qpos)
            dq_log.append(qvel)
            vdes_log.append(v_des)
            vcmd_log.append(v_cmd)
            hres_log.append(float(Hbm @ v_cmd))

            # Viewer sync
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            # Print debug
            if i % 500 == 0:
                print(f"Step {i} | Hbm@v_cmd={hres_log[-1]:.3e} | Base pos={base_pos:.3f}")

    # Convert logs
    q_log = np.array(q_log)
    dq_log = np.array(dq_log)
    vdes_log = np.array(vdes_log)
    vcmd_log = np.array(vcmd_log)
    hres_log = np.array(hres_log)
    timevec = np.arange(len(hres_log)) * dt

    # Plot diagnostics
    plt.figure(figsize=(10, 7))
    for j in range(4):
        plt.subplot(5, 1, j + 1)
        plt.plot(timevec, vdes_log[:, j], label=f"v_des j{j+1}")
        plt.plot(timevec, vcmd_log[:, j], label=f"v_cmd j{j+1}", linestyle='--')
        plt.ylabel(f"Joint {j+1}")
        plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(timevec, hres_log, label="Hbm @ v_cmd")
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Time [s]")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

new.py
import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# ---------- Cubic trajectory ----------
def cubic_trajectory(q0, qf, T, steps=200):
    t = np.linspace(0, T, steps)
    q = q0 + (3*(qf - q0)/T**2) * t**2 - (2*(qf - q0)/T**3) * t**3
    dq = (6*(qf - q0)/T**2) * t - (6*(qf - q0)/T**3) * t**2
    return t, q, dq

# ---------- Main Simulation ----------
def main():
    # Load model
    model = mujoco.MjModel.from_xml_path("new.xml")  # make sure XML path is correct
    data = mujoco.MjData(model)
   
    T = 10.0
    steps = 5000

    # Define joint trajectories
    q0, qf = 0.0, 1.5
    t, q_traj, dq_traj = cubic_trajectory(q0, qf, T, steps)
    _, q1_traj, dq1_traj = cubic_trajectory(0.0, 1.5, T, steps)
    _, q2_traj, dq2_traj = cubic_trajectory(0.0, 1.5, T, steps)
    _, q3_traj, dq3_traj = cubic_trajectory(0.0, 1.5, T, steps)

    # Actuator IDs
    vel1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel1")
    vel2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel2")
    vel3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel3")
    vel4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel4")

    # Logging
    q_log, dq_log, base_log, constraint_log = [], [], [], []

    paused = False
    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == ' ':
            paused = not paused

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        dt = model.opt.timestep
        for i in range(steps):
            step_start = time.time()

            # Desired joint velocities
            dq_des = np.array([dq_traj[i], dq1_traj[i], dq2_traj[i], dq3_traj[i]])

            # Compute mass matrix
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)

            # Extract base-manip coupling (row for base, cols for arms)
            Hbm = M[0:1, 1:5]   # base vs joints

            # Null space projection
            I = np.eye(4)
            Hbm_pinv = np.linalg.pinv(Hbm)
            N = I - Hbm_pinv @ Hbm
            v_cmd = N @ dq_des

            # Apply control
            data.ctrl[vel1_id] = v_cmd[0]
            data.ctrl[vel2_id] = v_cmd[1]
            data.ctrl[vel3_id] = v_cmd[2]
            data.ctrl[vel4_id] = v_cmd[3]

            if not paused:
                mujoco.mj_step(model, data)

            # Logs
            q_log.append(data.qpos.copy())
            dq_log.append(data.qvel.copy())
            base_log.append([data.qpos[0], data.qvel[0]])  # base position + velocity
            constraint_log.append((Hbm @ v_cmd).item())

            if i % 500 == 0:
                print(f"Step {i} | Base Pos={data.qpos[0]:.6f}, Base Vel={data.qvel[0]:.6f}, Hbm@V={ (Hbm@v_cmd).item():.6f}")

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # Convert logs
    q_log = np.array(q_log)
    dq_log = np.array(dq_log)
    base_log = np.array(base_log)
    constraint_log = np.array(constraint_log)

    # ---------- Plot Results ----------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Base motion
    axs[0].plot(base_log[:, 0], label="Base Pos")
    axs[0].plot(base_log[:, 1], label="Base Vel")
    axs[0].axhline(0, color="r", linestyle="--")
    axs[0].set_title("Base Motion")
    axs[0].legend()

    # Constraint satisfaction
    axs[1].plot(constraint_log, label="Hbm @ v_cmd")
    axs[1].axhline(0, color="r", linestyle="--")
    axs[1].set_title("Constraint Satisfaction")
    axs[1].legend()

    # Joint trajectories
    axs[2].plot(q_log[:, 1], label="Joint1")
    axs[2].plot(q_log[:, 2], label="Joint2")
    axs[2].plot(q_log[:, 3], label="Joint3")
    axs[2].plot(q_log[:, 4], label="Joint4")
    axs[2].plot(q_traj, 'k--', label="Desired Traj")
    axs[2].set_title("Manipulator Joint Positions")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

