import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# =====================================================
# Min-jerk trajectory
# =====================================================
def min_jerk(t, tf, q0, qf):
    tau = t / tf
    pos = q0 + (qf - q0) * (10*tau**3 - 15*tau**4 + 6*tau**5)
    vel = (qf - q0) * (30*tau**2 - 60*tau**3 + 30*tau**4) / tf
    return pos, vel

# =====================================================
# Load initial condition
# =====================================================
with open("init_state.pkl", "rb") as f:
    init = pickle.load(f)

q_start = init["q_start"]        # radians
q_goal  = init["q_target"]       # radians

# =====================================================
# MuJoCo setup
# =====================================================
XML_PATH = "nullspace.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

joints = ["joint1", "joint2", "joint3", "joint4"]
joint_ids = [model.joint(j).id for j in joints]

base_dof = model.joint("base_joint").dofadr[0]

dt = model.opt.timestep
T_FINAL = 3.0
steps = int(T_FINAL / dt)

# =====================================================
# Initialize state ONCE (critical)
# =====================================================
for i, jid in enumerate(joint_ids):
    data.qpos[jid] = q_start[i]

mujoco.mj_forward(model, data)

# =====================================================
# Logging
# =====================================================
time_log = []
base_pos = []
err_j1 = []
err_j2 = []

# =====================================================
# Run trajectory
# =====================================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(steps):
        t = step * dt

        pos_ref, vel_ref = min_jerk(t, T_FINAL, q_start, q_goal)

        # Base is passive → ctrl[0] untouched
        for i in range(4):
            data.ctrl[i] = vel_ref[i]

        mujoco.mj_step(model, data)

        # Logs
        time_log.append(t)
        base_pos.append(data.qpos[base_dof])

        q1 = data.qpos[joint_ids[0]]
        q2 = data.qpos[joint_ids[1]]

        err_j1.append(abs(q1))
        err_j2.append(abs(q2))

        viewer.sync()
        time.sleep(0.01)

# =====================================================
# Plot base motion
# =====================================================
plt.figure(figsize=(7,5))
plt.plot(time_log, base_pos, linewidth=3.5, label="Base Position (passive)")
plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("Base Position [rad]", fontsize=14, fontweight="bold")
plt.title("Passive Base Reaction (Min-Jerk Trajectory)", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# Plot joint errors
# =====================================================
plt.figure(figsize=(7,5))
plt.plot(time_log, err_j1, linewidth=3.5, label="Joint 1 Error")
plt.plot(time_log, err_j2, linewidth=3.5, label="Joint 2 Error")
plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("Error [rad]", fontsize=14, fontweight="bold")
plt.title("Joint Tracking Error (Min-Jerk)", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
