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
    """Return position and velocity at time t for a min-jerk trajectory."""
    tau = t / t_f
    pos = start + (goal - start) * (10*tau**3 - 15*tau**4 + 6*tau**5)
    vel = (goal - start) * (30*tau**2 - 60*tau**3 + 30*tau**4) / t_f
    return pos, vel

# -------------------------
# MuJoCo setup
# -------------------------
XML_PATH = "nullspace.xml"  # Your XML with velocity actuators
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Joint IDs
joints = ["joint1", "joint2", "joint3", "joint4"]
joint_ids = [model.joint(j).id for j in joints]

# Base joint index
base_dof = model.joint("base_joint").dofadr[0]

# -------------------------
# Trajectory parameters
# -------------------------
# Random initial for joint1 and joint2 between 0-10 deg
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

# -------------------------
# Run trajectory
# -------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(steps):
        t = step * dt

        # Compute min-jerk pos and vel
        pos, vel = min_jerk(t, t_final, q_start, q_goal)

        # Set joint positions directly (for reference)
        for i, jid in enumerate(joint_ids):
            data.qpos[jid] = np.radians(pos[i])

        # Set velocities to velocity actuators
        for i, jid in enumerate(joint_ids):
            data.ctrl[i] = np.radians(vel[i])  # velocity actuator expects rad/s

        # Step simulation
        mujoco.mj_step(model, data)

        # Log base position
        base_positions.append(data.qpos[base_dof])
        time_log.append(t)

        viewer.sync()
        time.sleep(0.01)

# -------------------------
# Plot base motion
# -------------------------
plt.figure(figsize=(6,4))
plt.plot(time_log, base_positions, label="Base Position")
plt.xlabel("Time [s]")
plt.ylabel("Base Position [rad/m]")
plt.title("Base Motion During Min-Jerk Trajectory")
plt.grid(True)
plt.legend()
plt.show()

