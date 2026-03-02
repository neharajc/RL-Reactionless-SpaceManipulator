import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# -------------------------
# Min-jerk trajectory function
# -------------------------
def min_jerk(t, t_f, start, goal):
    tau = t / t_f
    pos = start + (goal - start) * (10*tau**3 - 15*tau**4 + 6*tau**5)
    vel = (goal - start) * (30*tau**2 - 60*tau**3 + 30*tau**4) / t_f
    return pos, vel

# -------------------------
# Load initial state from pkl (IN RADIANS)
# -------------------------
with open("initial_state.pkl", "rb") as f:
    q_start = np.radians(pickle.load(f))

# -------------------------
# MuJoCo setup
# -------------------------
XML_PATH = "nullspace.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

joints = ["joint1", "joint2", "joint3", "joint4"]
joint_ids = [model.joint(j).id for j in joints]

base_dof = model.joint("base_joint").dofadr[0]
joint1_dof = model.joint("joint1").dofadr[0]
joint2_dof = model.joint("joint2").dofadr[0]


# INITIALIZATION 
data.qpos[0] = 0.0
data.qpos[1] = q_start[0]
data.qpos[2] = q_start[1]
data.qpos[3] = q_start[2]
data.qpos[4] = q_start[3]
data.qvel[:] = 0.0

mujoco.mj_forward(model, data)

print("Loaded initial state (rad):", q_start)

# -------------------------
# Trajectory parameters
# -------------------------
q_goal = np.zeros(4)
t_final = 3.0
dt = model.opt.timestep
steps = int(t_final / dt)


base_positions = []
time_log = []

trajpos, trajvel = [], []
trajposj2, trajvelj2 = [], []

qdes_j1, qdes_j2 = [], []
qact_j1, qact_j2 = [], []

vdes_j1, vdes_j2 = [], []
vact_j1, vact_j2 = [], []


# -------------------------
# Run trajectory
# -------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(steps):
        t = step * dt

        pos, vel = min_jerk(t, t_final, q_start, q_goal)

        trajpos.append(pos[0])
        trajvel.append(vel[0])
        trajposj2.append(pos[1])
        trajvelj2.append(vel[1])
        

        # velocity commands by skiping base motor
        for i in range(4):
            data.ctrl[i+1] = vel[i]

        mujoco.mj_step(model, data)

        qdes_j1.append(pos[0])
        qdes_j2.append(pos[1])
        qact_j1.append(data.qpos[joint1_dof])
        qact_j2.append(data.qpos[joint2_dof])
        
        vdes_j1.append(vel[0])
        vdes_j2.append(vel[1])
        vact_j1.append(data.qvel[joint1_dof])
        vact_j2.append(data.qvel[joint2_dof])


        base_positions.append(data.qpos[base_dof])
        time_log.append(t)

        viewer.sync()
        time.sleep(0.01)
        

# -------------------------
# Plot base motion
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(time_log, base_positions, linewidth=3.5, label="Base Position (passive)")
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
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(time_log, trajvel, linewidth=3.5, label="required vel joint1")
plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("joint 1 velocity ", fontsize=14, fontweight="bold")
plt.title("joint1 velocity",
          fontsize=16, fontweight="bold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(7,5))
plt.plot(time_log, trajpos, linewidth=3.5, label="required position joint1")
plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("joint 1 position ", fontsize=14, fontweight="bold")
plt.title("joint1 position",
          fontsize=16, fontweight="bold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(time_log, trajvelj2, linewidth=3.5, label="required vel joint2")
plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("joint 2 velocity ", fontsize=14, fontweight="bold")
plt.title("joint2 velocity",
          fontsize=16, fontweight="bold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(time_log, trajposj2, linewidth=3.5, label="required pos joint2")
plt.xlabel("Time [s]", fontsize=14, fontweight="bold")
plt.ylabel("joint 2 position ", fontsize=14, fontweight="bold")
plt.title("joint2 position",
          fontsize=16, fontweight="bold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
plt.plot(time_log, vdes_j1, 'k--', label="Desired Vel J1")
plt.plot(time_log, vact_j1, 'r',  label="Actual Vel J1")
plt.plot(time_log, vdes_j2, 'k:', label="Desired Vel J2")
plt.plot(time_log, vact_j2, 'b',  label="Actual Vel J2")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [rad/s]")
plt.title("Desired vs Actual Joint Velocities")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Plot desired vs actual
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(time_log, qdes_j1, 'k--', label="Desired J1")
plt.plot(time_log, qact_j1, 'r', label="Actual J1")
plt.plot(time_log, qdes_j2, 'k:', label="Desired J2")
plt.plot(time_log, qact_j2, 'b', label="Actual J2")
plt.xlabel("Time [s]")
plt.ylabel("Position [rad]")
plt.title("Desired vs Actual Joint Positions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
