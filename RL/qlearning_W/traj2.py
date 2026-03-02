import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# -------------------------
# Min-jerk trajectory function
# -------------------------
def min_jerk(t, t_f, start, goal):
    tau = t / t_f
    pos = start + (goal - start) * (10*tau**3 - 15*tau**4 + 6*tau**5)
    vel = (goal - start) * (30*tau**2 - 60*tau**3 + 30*tau**4) / t_f
    return pos, vel

# -------------------------
# MuJoCo setup
# -------------------------
XML_PATH = "nullspace.xml"  # XML with base passive, joints velocity actuated
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Joint IDs (skip base)
joints = ["joint1", "joint2", "joint3", "joint4"]
joint_ids = [model.joint(j).id for j in joints]

# Base DOF index (passive)
base_dof = model.joint("base_joint").dofadr[0]

# -------------------------
# Trajectory parameters
# -------------------------
q_start = np.array([random.uniform(0,10),
                    random.uniform(0,10),
                    0.0,
                    0.0])
q_goal  = np.array([0.0, 0.0, 0.0, 0.0])  # degrees
t_final = 3.0  # seconds
dt = model.opt.timestep
steps = int(t_final / dt)

# Data storage
base_positions = []
time_log = []
error_joint1 = []
error_joint2 = []

# -------------------------
# Run trajectory
# -------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(steps):
        t = step * dt

        # Compute min-jerk pos and vel
        pos, vel = min_jerk(t, t_final, q_start, q_goal)

        # Set joint positions (for reference)
        for i, jid in enumerate(joint_ids):
            data.qpos[jid] = np.radians(pos[i])

        # Set velocities to velocity actuators (joints only)
        for i in range(len(joint_ids)):
            data.ctrl[i] = np.radians(vel[i])

        # Step simulation
        mujoco.mj_step(model, data)

        # Log base position (passive)
        base_positions.append(data.qpos[base_dof])
        time_log.append(t)

        # Log errors for joint1 and joint2
        current_joint1 = np.degrees(data.qpos[joint_ids[0]])
        current_joint2 = np.degrees(data.qpos[joint_ids[1]])
        error_joint1.append(abs(current_joint1 - q_goal[0]))
        error_joint2.append(abs(current_joint2 - q_goal[1]))

        viewer.sync()
        time.sleep(0.01)

# -------------------------
# -------------------------
# Plot base motion (passive)
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(
    time_log,
    base_positions,
    linewidth=3.5,
    label="Base Position (passive)"
)

plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("Base Position [rad]", fontsize=14, fontweight="bold")
plt.title(
    "Passive Base Motion During Min-Jerk Trajectory",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------
# Plot joint tracking errors
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(
    time_log,
    error_joint1,
    linewidth=3.5,
    label="Joint 1 Error"
)
plt.plot(
    time_log,
    error_joint2,
    linewidth=3.5,
    label="Joint 2 Error"
)

plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("Error [deg]", fontsize=14, fontweight="bold")
plt.title(
    "Joint Tracking Error During Min-Jerk Trajectory",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


