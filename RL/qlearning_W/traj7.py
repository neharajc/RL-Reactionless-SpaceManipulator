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
# Load initial state from pkl
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
data.qpos[0] = 0.0  
data.qpos[1]=np.radians(q_start[0])
data.qpos[2]=np.radians(q_start[1])
data.qpos[3]=np.radians(q_start[2])
data.qpos[4]=np.radians(q_start[3])
mujoco.mj_forward(model, data)
print(data.qpos[1])

# -------------------------
# Trajectory parameters
# -------------------------
q_goal  = np.array([0.0, 0.0, 0.0, 0.0]) 
t_final = 3.0  
dt = model.opt.timestep
steps = int(t_final / dt)

# Data storage
base_positions = []
time_log = []

error_joint1 = []
error_joint2 = []

trajpos=[]
trajvel=[]
trajposj2=[]
trajvelj2=[]

qdes_j1 = []
qdes_j2 = []
qact_j1 = []
qact_j2 = []

print("Loaded initial state (deg):", q_start)

# -------------------------
# Run trajectory
# -------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(steps):
        t = step * dt

        # Compute min-jerk pos and vel
        pos, vel = min_jerk(t, t_final, q_start, q_goal)

        trajpos.append(pos[0])
        trajvel.append(vel[0])
        trajposj2.append(pos[1])
        trajvelj2.append(vel[1])

        # Apply velocities to actuators (joints only)
        for i in range(len(joint_ids)):
            data.ctrl[i+1] = vel[i]

        mujoco.mj_step(model, data)

        # -------- CORRECTLY INDENTED LOGGING --------
        qdes_j1.append(pos[0])
        qdes_j2.append(pos[1])

        qact_j1.append(data.qpos[joint_ids[0]])
        qact_j2.append(data.qpos[joint_ids[1]])

        # Log base position
        base_positions.append(data.qpos[base_dof])
        time_log.append(t)

        viewer.sync()
        time.sleep(0.01)

        # Log joint errors
        """current_joint1 = np.degrees(data.qpos[joint_ids[0]])
        current_joint2 = np.degrees(data.qpos[joint_ids[1]])
        error_joint1.append(abs(current_joint1 -  0))
        error_joint2.append(abs(current_joint2 - 0))"""


      

        

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

"""
# -------------------------
# Plot joint tracking errors
# -------------------------
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
plt.show()""" 
plt.figure(figsize=(8,5))
plt.plot(time_log, qdes_j1, 'k--', label="Desired J1")
plt.plot(time_log, qact_j1, 'r',  label="Actual J1")
plt.plot(time_log, qdes_j2, 'k:', label="Desired J2")
plt.plot(time_log, qact_j2, 'b',  label="Actual J2")
plt.xlabel("Time [s]")
plt.ylabel("Position [rad]")
plt.title("Desired vs Actual Joint Positions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
