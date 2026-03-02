import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# ---------- Cubic trajectory ----------
def cubic_trajectory(q0, qf, T, steps=200):
    t = np.linspace(0, T, steps)
    q = q0 + (3*(qf - q0)/T**2) * t**2 - (2*(qf - q0)/T**3) * t**3
    dq = (6*(qf - q0)/T**2) * t - (6*(qf - q0)/T**3) * t**2
    return t, q, dq

# ---------- Velocity optimizer using SciPy ----------
def optimize_two_joints_scipy(h, v_des, cmd_idx=[0,1], free_idx=[2,3]):
    """
    Optimize velocities:
    - cmd_idx: indices of commanded joints (v_des)
    - free_idx: indices of free joints
    - h: constraint vector h1..h4
    - v_des: desired velocities for commanded joints
    Returns full 4-element v_cmd
    """
    n = len(h)
    v0 = np.zeros(n)
    v0[cmd_idx] = v_des       # initial guess

    # Cost: minimize squared error from desired
    def cost(v):
        return np.sum((v - v0)**2)

    # Constraint: h^T v = 0
    cons = {'type': 'eq', 'fun': lambda v: np.dot(h, v)}

    res = minimize(cost, v0, constraints=cons, method='SLSQP')
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x

# ---------- Main Simulation ----------
def main():
    model = mujoco.MjModel.from_xml_path("try.xml")
    data = mujoco.MjData(model)
        
    T = 10.0
    steps = 5000
    dt = model.opt.timestep

    # Cubic trajectories for joint1 & joint2
    t, q1_traj, dq1_traj = cubic_trajectory(0.0, 1.5, T, steps)
    _, q2_traj, dq2_traj = cubic_trajectory(0.0, 1.5, T, steps)

    # --- Actuator mapping ---
    print("\n--- Actuator mapping (ctrl → joint) ---")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
        print(f"ctrl[{i}] actuator '{name}' → joint '{jname}'")

    vel_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vel{i+1}") for i in range(4)]

    paused = False
    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == ' ':
            paused = not paused

    # Logs
    q_log, dq_log, q_des_log, vcmd_log, constraint_log = [], [], [], [], []

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        M = np.zeros((model.nv, model.nv))
        for i in range(steps):
            step_start = time.time()

            # Desired velocities and positions for joints 1 & 2
            dq_des = np.array([dq1_traj[i], dq2_traj[i]])
            q_des = np.array([q1_traj[i], q2_traj[i]])
           
            # Step physics
            if not paused:
                mujoco.mj_step(model, data)

            # Mass matrix and constraint vector (adjust indices for your robot)
            mujoco.mj_fullM(model, M, data.qM)
            Hbm = M[0, 1:5]  # first row, joints 1-4
            h = Hbm.flatten()
            
            # Compute commanded velocities
            v_cmd = optimize_two_joints_scipy(h, dq_des, cmd_idx=[0,1], free_idx=[2,3])

            # Apply commands
            for j, aid in enumerate(vel_ids):
                data.ctrl[aid] = float(v_cmd[j])

            # Logging
            q_log.append(data.qpos.copy())
            dq_log.append(data.qvel.copy())
            q_des_log.append([q1_traj[i], q2_traj[i], 0.0, 0.0])
            vcmd_log.append(v_cmd)
            constraint_log.append(np.dot(h, v_cmd))

            # Print every 500 steps
            if i % 1 == 0:
                print(f"\nStep {i}")
                print("qpos:", data.qpos)
                print("qvel:", data.qvel)
                print("q_des:", q_des)
                print("v_cmd:", v_cmd)
                print(f"h · v_cmd = {constraint_log[-1]:.6f}")

            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # Convert logs to arrays
    q_log = np.array(q_log)
    dq_log = np.array(dq_log)
    q_des_log = np.array(q_des_log)
    vcmd_log = np.array(vcmd_log)
    time_vec = np.linspace(0, T, steps)

    # --- Plot Joint 2: Desired vs Commanded vs Actual Velocity ---
    plt.figure(figsize=(10, 6))
    plt.plot(time_vec, dq2_traj, 'k--', label="Joint2 Desired Vel")
    plt.plot(time_vec, [vc[1] for vc in vcmd_log], 'r-', label="Joint2 Commanded Vel")
    plt.plot(time_vec, dq_log[:, 2], 'b', label="Joint2 Actual Vel")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [rad/s]")
    plt.title("Joint 2: Desired vs Commanded vs Actual Velocity")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
