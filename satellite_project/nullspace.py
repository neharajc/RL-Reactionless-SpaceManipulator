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
    model = mujoco.MjModel.from_xml_path("nullspace.xml")
    data = mujoco.MjData(model)

    T = 10.0
    steps = 5000

    # Trajectories
    q, qf = 0.0, 1.5
    q1, qf1 = 0.0, 1.5
    q2, qf2 = 0.0, 1.5
    q3, qf3 = 0.0, 1.5
    t, q_traj, dq_traj = cubic_trajectory(q, qf, T, steps)
    t1, q1_traj, dq1_traj = cubic_trajectory(q1, qf1, T, steps)
    t2, q2_traj, dq2_traj = cubic_trajectory(q2, qf2, T, steps)
    t3, q3_traj, dq3_traj = cubic_trajectory(q3, qf3, T, steps)

    # Actuator IDs
    vel1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel1")
    vel2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel2")
    vel3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel3")
    vel4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel4")
    motor0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor0")
    print("Actuator IDs:", vel1_id, vel2_id, vel3_id, vel4_id, motor0_id)

    # Logging
    q_log, dq_log, tau_log = [], [], []

    # Pause toggle
    paused = False

    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == " ":
            paused = not paused

    # Safety / controller params (tune these)
    k_fb = 0.1          # feedback gain on base velocity (try 0.01..1)
    eps = 1e-9              # small regularizer for denom
    max_joint_vel = 8.0     # clamp for commanded joint velocities (rad/s)
    max_delta_norm = 3.0    # clamp for correction vector norm
    kp_base = 6.0           # small PD on base motor (backup)
    kd_base = 1.0

    # Which DOFs are manipulator (arm) columns? assume base is DOF 0, arm are 1..nv-1
    nv = model.nv
    base_dof = 0
    arm_dofs = list(range(1, nv))   # expects 4 arm dofs

    # Viewer loop
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        i = 0
        dt = model.opt.timestep
        Mfull = np.zeros((model.nv, model.nv))

        while viewer.is_running() and i < steps:
            step_start = time.time()

            # Desired state
            q_des, dq_des = q_traj[i], dq_traj[i]
            q_des1, dq_des1 = q1_traj[i], dq1_traj[i]
            q_des2, dq_des2 = q2_traj[i], dq2_traj[i]
            q_des3, dq_des3 = q3_traj[i], dq3_traj[i]

            # Integrate physics first (keeps behavior same as your original loop)
            if not paused:
                mujoco.mj_step(model, data)

            # Debug print of qpos
            print(f"print qpos: {data.qpos}")

            # Compute full mass matrix
            mujoco.mj_fullM(model, Mfull, data.qM)

            # Extract Hbm: base row x arm columns (shape 1x4)
            Hbm = Mfull[np.ix_([base_dof], arm_dofs)].astype(float)  # (1,4)
            print(f"hbm at step {i}: \n {Hbm} \n")

            # Build desired manipulator velocity vector
            v_des = np.array([dq_des, dq_des1, dq_des2, dq_des3], dtype=float)

            # Measure base velocity
            v_b = float(data.qvel[base_dof])

            # Compute scalar residual r = -Hbm @ v_des - k_fb * v_b
            hbmvdes = float(Hbm @ v_des)   # scalar
            r = -hbmvdes - k_fb * v_b      # scalar

            # Compute pseudoinverse safely (1x4 -> 4x1)
            denom = float(Hbm @ Hbm.T)     # scalar
            if denom <= eps:
                # coupling negligible -> no correction
                delta_q = np.zeros(4, dtype=float)
            else:
                Hbm_pinv = (Hbm.T / (denom + 0.0)).reshape(4,)   # (4,)
                delta_q = Hbm_pinv * r                          # (4,)

                # Safety limiting: per-element clip and global norm clip
                delta_q = np.clip(delta_q, -max_joint_vel, max_joint_vel)
                norm_delta = np.linalg.norm(delta_q)
                if norm_delta > max_delta_norm:
                    delta_q = delta_q * (max_delta_norm / norm_delta)

            # Final commanded velocities (clamped)
            v_cmd = np.clip(v_des + delta_q, -max_joint_vel, max_joint_vel)

            # Diagnostic prints every 200 steps
            if i % 200 == 0 or i == steps - 1:
                try:
                    denom_val = float(Hbm @ Hbm.T)
                except Exception:
                    denom_val = np.nan
                print(
                    "step",
                    i,
                    "base_qpos",
                    float(data.qpos[base_dof]),
                    "base_qvel",
                    float(data.qvel[base_dof]),
                    "Hbm@vdes",
                    hbmvdes,
                    "Hbm@vcmd",
                    float(Hbm @ v_cmd),
                    "denom",
                    denom_val,
                    "motor0_ctrl",
                    float(data.ctrl[motor0_id]),
                )

            # Small PD on base motor (numerical safeguard). If you prefer pure null-space, set to 0.
            base_pos = float(data.qpos[base_dof])
            base_vel = float(data.qvel[base_dof])
            data.ctrl[motor0_id] = float(-kp_base * base_pos - kd_base * base_vel)

            # Send velocity targets to actuators
            data.ctrl[vel1_id] = float(v_cmd[0])
            data.ctrl[vel2_id] = float(v_cmd[1])
            data.ctrl[vel3_id] = float(v_cmd[2])
            data.ctrl[vel4_id] = float(v_cmd[3])

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

    # (optional plotting omitted to keep script identical in structure)

if __name__ == "__main__":
    main()

