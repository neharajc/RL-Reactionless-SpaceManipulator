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

# ---------- Helpers ----------
def dof_of_actuator(model, act_id):
    # Map actuator -> joint -> DoF index (assumes hinge/slide joint, 1 DoF)
    j = model.actuator_trnid[act_id, 0]
    idx = np.nonzero(model.dof_jntid == j)[0]
    return int(idx[0])

def build_projector(Hbm, lam=1e-4):
    # Damped projector: N = I - H^T (H H^T + lam^2 I)^-1 H
    # Handles arbitrary row count; for 1-DoF base, G is scalar
    m = Hbm.shape[0]
    G = (Hbm @ Hbm.T) + (lam**2) * np.eye(m)
    # Solve instead of invert for numerical stability
    N = np.eye(Hbm.shape[1]) - Hbm.T @ np.linalg.solve(G, Hbm)
    return N, G

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
        trnid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
        print(f"ctrl[{i}] actuator '{name}' → joint '{jname}'")

    # Trajectory
    steps = 5000
    T = 10.0
    q0, qf = 0.0, 1.5
    q1, qf1 = 0.0, 1.5
    q2, qf2 = 0.0, 1.5
    q3, qf3 = 0.0, 1.5

    t,  q_traj,  dq_traj  = cubic_trajectory(q0,  qf,  T, steps)
    t1, q1_traj, dq1_traj = cubic_trajectory(q1, qf1, T, steps)
    t2, q2_traj, dq2_traj = cubic_trajectory(q2, qf2, T, steps)
    t3, q3_traj, dq3_traj = cubic_trajectory(q3, qf3, T, steps)

    # Actuator IDs (velocity servos + base motor)
    vel1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel1")
    vel2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel2")
    vel3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel3")
    vel4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel4")
    motor0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor0")
    print("Actuator IDs:", vel1_id, vel2_id, vel3_id, vel4_id, motor0_id)

    # ---- DoF indices for base and arm (robust mapping by actuators) ----
    vel_ids  = [vel1_id, vel2_id, vel3_id, vel4_id]
    arm_dofs = np.array([dof_of_actuator(model, a) for a in vel_ids], dtype=int)

    base_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_joint")
    base_rows = np.nonzero(model.dof_jntid == base_jid)[0]   # length 1 for hinge
    if base_rows.size == 0:
        print("Warning: No base DoF found; projection will be identity.")
    # Initialize derived quantities once so data.qM is valid
    mujoco.mj_forward(model, data)

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

            # Desired velocities for arm joints (order matches vel1..vel4)
            vdes = np.array([dq_traj[i], dq1_traj[i], dq2_traj[i], dq3_traj[i]], dtype=float)

            # Dense mass matrix
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)

            # Hbm block: base row(s) × arm columns (1x4 for base hinge)
            if base_rows.size > 0:
                Hbm = M[np.ix_(base_rows, arm_dofs)]  # shape (1, 4)
                # Damped projector
                N, G = build_projector(Hbm, lam=1e-4)
                # Optional feedback to kill residual base reaction with measured v
                v_arm_meas = data.qvel[arm_dofs].copy()
                r = Hbm @ v_arm_meas                   # residual base momentum rate proxy
                vcmd = (N @ vdes) - (Hbm.T @ np.linalg.solve(G, r))
            else:
                vcmd = vdes

            # Apply controls: keep base actuator passive (ctrl=0)
            data.ctrl[motor0_id] = 0.0
            data.ctrl[vel1_id] = float(vcmd[0])
            data.ctrl[vel2_id] = float(vcmd[1])
            data.ctrl[vel3_id] = float(vcmd[2])
            data.ctrl[vel4_id] = float(vcmd[3])

            if not paused:
                mujoco.mj_step(model, data)

            # Optional debug print every 100 steps
            if i % 100 == 0:
                print(f"step {i} qpos: {data.qpos}")

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

    # Convert logs
    q_log = np.array(q_log)
    dq_log = np.array(dq_log)
    tau_log = np.array(tau_log)

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    for j in range(model.nq):
        plt.plot(t, q_log[:, j], label=f'Joint {j}')
    plt.plot(t,  q_traj,  'k--', label='Desired Joint1')
    plt.plot(t1, q1_traj, 'r--', label='Desired Joint2')
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

