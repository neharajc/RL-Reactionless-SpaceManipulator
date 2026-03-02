import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# =========================================================
# Min-jerk trajectory (radians)
# =========================================================
def min_jerk(t, t_f, q0, qf):
    tau = t / t_f
    pos = q0 + (qf - q0) * (10*tau**3 - 15*tau**4 + 6*tau**5)
    vel = (qf - q0) * (30*tau**2 - 60*tau**3 + 30*tau**4) / t_f
    return pos, vel


# =========================================================
# Load initial condition
# =========================================================
with open("init_state.pkl", "rb") as f:
    init_state = pickle.load(f)

q_start = init_state["q_start"]     # radians
q_goal  = init_state["q_target"]    # radians

print("Initial state (deg):", np.degrees(q_start))
print("Target state (deg):", np.degrees(q_goal))


# =========================================================
# MuJoCo setup
# =========================================================
XML_PATH = "nullspace.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Joint names (base is passive, index 0 actuator)
joint_names = ["joint1", "joint2", "joint3", "joint4"]
joint_ids = [model.joint(j).id for j in joint_names]

# Base DOF (passive)
base_dof = model.joint("base_joint").dofadr[0]


# =========================================================
# Trajectory parameters
# =========================================================
t_final = 3.0
dt = model.opt.timestep
steps = int(t_final / dt)

# Logs
time_log = []
base_positions = []
error_joint1 = []
error_joint2 = []


# =========================================================
# Run trajectory
# =========================================================
with mujoco.viewer.launch_passive(model, data) as viewer:

    # Initialize joints
    for i, jid in enumerate(joint_ids):
        data.qpos[jid] = q_start[i]
    mujoco.mj_forward(model, data)

    for step in range(steps):
        t = step * dt

        # Min-jerk reference
        q_ref, qd_ref = min_jerk(t, t_final, q_start, q_goal)

        # -------------------------------------------------
        # Apply velocities (ctrl[0] = base motor → unused)
        # -------------------------------------------------
        data.ctrl[0] = 0.0
        for i in range(4):
            data.ctrl[i+1] = qd_ref[i]

        mujoco.mj_step(model, data)

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------
        time_log.append(t)
        base_positions.append(data.qpos[base_dof])

        j1 = np.degrees(data.qpos[joint_ids[0]])
        j2 = np.degrees(data.qpos[joint_ids[1]])

        error_joint1.append(abs(j1))
        error_joint2.append(abs(j2))

        viewer.sync()
        time.sleep(0.01)


# =========================================================
# Plot base motion
# =========================================================
plt.figure(figsize=(7,5))
plt.plot(time_log, base_positions, linewidth=3.5,
         label="Base Position (passive)")

plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("Base Position [rad]", fontsize=14, fontweight="bold")
plt.title("Passive Base Motion During Min-Jerk Trajectory",
          fontsize=16, fontweight="bold")

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# Plot joint tracking error
# =========================================================
plt.figure(figsize=(7,5))
plt.plot(time_log, error_joint1, linewidth=3.5, label="Joint 1 Error")
plt.plot(time_log, error_joint2, linewidth=3.5, label="Joint 2 Error")

plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("Error [deg]", fontsize=14, fontweight="bold")
plt.title("Joint Tracking Error During Min-Jerk Trajectory",
          fontsize=16, fontweight="bold")

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
