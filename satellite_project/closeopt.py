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

# ---------- Null-space optimization with Lagrange multipliers ----------
def compute_vcmd(Hbm, dq_des, reg=1e-12):
    h1, h2 = Hbm[0, 0], Hbm[0, 1]
    A = np.array([
        [2, 0, h1],
        [0, 2, h2],
        [h1, h2, reg]
    ])
    b = np.array([2*dq_des[0], 2*dq_des[1], 0])
    try:
        x = np.linalg.solve(A, b)
        return x[0], x[1]
    except np.linalg.LinAlgError:
        return dq_des[0], dq_des[1]

# ---------- Main Simulation ----------
def main():
    model = mujoco.MjModel.from_xml_path("try.xml")
    data = mujoco.MjData(model)
        
    T = 10.0
    steps = 5000
    # Cubic trajectories for joint1 & joint2
    t, q1_traj, dq1_traj = cubic_trajectory(0.0, 1.5, T, steps)
    _, q2_traj, dq2_traj = cubic_trajectory(0.0, 1.5, T, steps)
    # --- Actuator mapping check ---
    print("\n--- Actuator mapping (ctrl → joint) ---")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
        print(f"ctrl[{i}] actuator '{name}' → joint '{jname}'")
    # Actuator IDs
    vel1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel1")
    vel2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel2")
    vel3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel3")
    vel4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vel4")
    # Joint indexes
    joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    joint2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    paused = False
    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == ' ':
            paused = not paused
    # Logs
    q_log, dq_log, q_des_log, vcmd_log, constraint_log = [], [], [], [], []
    # --- Closed Loop gain ---
    Kp = 3.0  # Tune as needed: 1.0 ~ 5.0 is reasonable
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        dt = model.opt.timestep
        for i in range(steps):
            step_start = time.time()
            # Desired velocities and positions
            dq_des = np.array([dq1_traj[i], dq2_traj[i]])
            q_des = np.array([q1_traj[i], q2_traj[i]])
            # Read actual joint positions
            q1_act = float(data.qpos[joint1_id])
            q2_act = float(data.qpos[joint2_id])
            # Closed-loop correction
            v1_cmd_fb = dq1_traj[i] + Kp * (q1_traj[i] - q1_act)
            v2_cmd_fb = dq2_traj[i] + Kp * (q2_traj[i] - q2_act)
            # Step physics
            if not paused:
                mujoco.mj_step(model, data)
            # Mass matrix for constraint log (optional)
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)
            Hbm = M[0:1, 1:3]
            # Optional: use null-space optimization here (update as needed)
            # v1_cmd_opt, v2_cmd_opt = compute_vcmd(Hbm, np.array([v1_cmd_fb, v2_cmd_fb]))
            # Apply commands to actuators
            data.ctrl[vel1_id] = v1_cmd_fb
            data.ctrl[vel2_id] = v2_cmd_fb
            # leave vel3, vel4 free (don’t force them)
            # Logging
            q_log.append(data.qpos.copy())     
            dq_log.append(data.qvel.copy())    
            q_des_log.append([q1_traj[i], q2_traj[i], 0.0, 0.0])
            vcmd_log.append([v1_cmd_fb, v2_cmd_fb])
            constraint_log.append((Hbm @ np.array([v1_cmd_fb, v2_cmd_fb])).item())
            # Print some values
            if i % 500 == 0:  
                print(f"\nStep {i}")
                print("qpos:", data.qpos)
                print("qvel:", data.qvel)
                print("q_des:", q_des)
                print("v_cmd:", [v1_cmd_fb, v2_cmd_fb])
                print(f"Hbm @ v_cmd = {constraint_log[-1]:.6f}")
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    # Convert logs
    q_log = np.array(q_log)
    dq_log = np.array(dq_log)
    q_des_log = np.array(q_des_log)
    vcmd_log = np.array(vcmd_log)
    # --- Plot Achieved vs Desired ---
    time_vec = np.linspace(0, T, steps)
    plt.figure(figsize=(10, 6))
    plt.plot(time_vec, q_log[:, joint1_id], label="Joint1 Achieved")
    plt.plot(time_vec, q_log[:, joint2_id], label="Joint2 Achieved")
    plt.plot(time_vec, q_des_log[:, 0], 'k--', label="Joint1 Desired")
    plt.plot(time_vec, q_des_log[:, 1], 'r--', label="Joint2 Desired")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [rad]")
    plt.legend()
    plt.title("Achieved vs Desired Joint Positions")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
