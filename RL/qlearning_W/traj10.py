import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# -------------------------
# Min-jerk trajectory
# -------------------------
def min_jerk(t, t_f, start, goal):
    tau = np.clip(t / t_f, 0, 1)
    pos = start + (goal - start) * (10*tau**3 - 15*tau**4 + 6*tau**5)
    vel = (goal - start) * (30*tau**2 - 60*tau**3 + 30*tau**4) / t_f
    return pos, vel

# -------------------------
# Load initial state (deg)
# -------------------------
with open("initial_state.pkl", "rb") as f:
    q_start = pickle.load(f)   # degrees

# -------------------------
# MuJoCo setup
# -------------------------
XML_PATH = "nullspace.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Joints (skip base)
joints = ["joint1", "joint2", "joint3", "joint4"]
joint_ids = [model.joint(j).id for j in joints]

# Base DOF index
base_dof = model.joint("base_joint").dofadr[0]

# -------------------------
# Initialize state
# -------------------------
for i in range(4):
    data.qpos[i] = np.radians(q_start[i])

data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# -------------------------
# Trajectory params
# -------------------------
q_goal  = np.array([0.0, 0.0, 0.0, 0.0])  # deg
t_final = 3.0
dt = model.opt.timestep
steps = int(t_final / dt)

# -------------------------
# Logs
# -------------------------
time_log = []

des_pos = []
des_vel = []

act_pos = []
act_vel = []

torques = []

base_positions = []

error_pos = []
error_vel = []

# -------------------------
# Run simulation
# -------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(steps):
        t = step * dt

        # Desired trajectory
        pos, vel = min_jerk(t, t_final, q_start, q_goal)

        # Command velocities (rad/s)
        for i in range(len(joint_ids)):
            data.ctrl[i] = np.radians(vel[i])

        # Step
        mujoco.mj_step(model, data)

        # Log time
        time_log.append(t)

        # Desired
        des_pos.append(pos[0])
        des_vel.append(vel[0])

        # Actual (rad → deg)
        q_actual = np.degrees(data.qpos[joint_ids[0]])
        v_actual = np.degrees(data.qvel[joint_ids[0]])

        act_pos.append(q_actual)
        act_vel.append(v_actual)

        # Torque (if available)
        torques.append(data.qfrc_actuator[joint_ids[0]])

        # Base
        base_positions.append(data.qpos[base_dof])

        # Errors
        error_pos.append(abs(q_actual - pos[0]))
        error_vel.append(abs(v_actual - vel[0]))

        viewer.sync()
        time.sleep(0.01)

# -------------------------
# PLOTS
# -------------------------

# Position tracking
plt.figure(figsize=(7,5))
plt.plot(time_log, des_pos, label="Desired Position")
plt.plot(time_log, act_pos, label="Actual Position", linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Joint 1 Position [deg]")
plt.title("Joint 1 Position Tracking")
plt.legend()
plt.grid()
plt.show()

# Velocity tracking
plt.figure(figsize=(7,5))
plt.plot(time_log, des_vel, label="Desired Velocity")
plt.plot(time_log, act_vel, label="Actual Velocity", linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Joint 1 Velocity [deg/s]")
plt.title("Joint 1 Velocity Tracking")
plt.legend()
plt.grid()
plt.show()

# Torque
plt.figure(figsize=(7,5))
plt.plot(time_log, torques)
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.title("Joint 1 Actuator Torque")
plt.grid()
plt.show()

# Position error
plt.figure(figsize=(7,5))
plt.plot(time_log, error_pos)
plt.xlabel("Time [s]")
plt.ylabel("Position Error [deg]")
plt.title("Joint 1 Position Error")
plt.grid()
plt.show()

# Velocity error
plt.figure(figsize=(7,5))
plt.plot(time_log, error_vel)
plt.xlabel("Time [s]")
plt.ylabel("Velocity Error [deg/s]")
plt.title("Joint 1 Velocity Error")
plt.grid()
plt.show()

# Base motion
plt.figure(figsize=(7,5))
plt.plot(time_log, base_positions)
plt.xlabel("Time [s]")
plt.ylabel("Base Position [rad]")
plt.title("Passive Base Motion")
plt.grid()
plt.show()
