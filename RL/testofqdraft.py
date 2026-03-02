import mujoco
import mujoco.viewer
import numpy as np
import pickle
from collections import defaultdict
import time

# -------------------------
# Load saved Q-table
# -------------------------
with open("q_table_81.pkl", "rb") as f:
    Q_loaded = pickle.load(f)

Q = defaultdict(float, Q_loaded)

# -------------------------
# Action definitions
# -------------------------
TORQUE_LEVELS = [-2.5, 0.0, 2.5]
ACTIONS = [(t1, t2, t3, t4)
           for t1 in TORQUE_LEVELS
           for t2 in TORQUE_LEVELS
           for t3 in TORQUE_LEVELS
           for t4 in TORQUE_LEVELS]

# -------------------------
# Discretisation
# -------------------------
ANGLE_MIN, ANGLE_MAX, STEP_DEG = 0.0, 10.0, 0.5
BINS = int(round((ANGLE_MAX - ANGLE_MIN) / STEP_DEG))

def angle_to_index(angle):
    a = max(min(angle, ANGLE_MAX), ANGLE_MIN)
    idx = int((a - ANGLE_MIN) / STEP_DEG)
    return max(0, min(BINS - 1, idx))

# -------------------------
# Choose best action (greedy)
# -------------------------
def choose_best_action(state_key):
    qvals = [Q[(state_key, idx)] for idx in range(len(ACTIONS))]
    return int(np.argmax(qvals))

# -------------------------
# Load MuJoCo model
# -------------------------
model = mujoco.MjModel.from_xml_path("nullspace.xml")
data = mujoco.MjData(model)

# Joint IDs
j1_id = model.joint("joint1").id
j2_id = model.joint("joint2").id
j3_id = model.joint("joint3").id
j4_id = model.joint("joint4").id

# Base joint DOF (HINGE about Z)
base_joint_id = model.joint("base_joint").id
base_dof = model.joint("base_joint").dofadr[0]

print("Base joint id :", base_joint_id)
print("Base DOF index:", base_dof)
print("Base joint type (0=free,1=ball,2=slide,3=hinge):",
      model.jnt_type[base_joint_id])

# -------------------------
# Test function
# -------------------------
def test_agent(steps=800, target1=0.0, target2=0.0):

    mujoco.mj_resetData(model, data)

    # Random initial joint positions
    data.qpos[j1_id] = np.radians(np.random.uniform(0, 10))
    data.qpos[j2_id] = np.radians(np.random.uniform(0, 10))
    mujoco.mj_forward(model, data)

    base_vel_history = []

    print("\n========== TEST START ==========")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(steps):

            # Joint angles (degrees)
            j1 = np.degrees(data.qpos[j1_id])
            j2 = np.degrees(data.qpos[j2_id])
            j3 = np.degrees(data.qpos[j3_id])
            j4 = np.degrees(data.qpos[j4_id])

            # Base angular velocity (rad/s)
            base_qvel = data.qvel[base_dof]
            base_vel_history.append(abs(base_qvel))

            print(
                f"Step {step:03d} | "
                f"j1={j1:6.3f}°, j2={j2:6.3f}°, "
                f"j3={j3:6.3f}°, j4={j4:6.3f}° | "
                f"base_qvel(z-rot)={base_qvel:+.6f} rad/s"
            )

            # Check target reaching
            err = abs(j1 - target1) + abs(j2 - target2)
            if err < 0.5:
                print("🎉 TARGET REACHED!")
                break

            # State
            s_key = (
                angle_to_index(j1),
                angle_to_index(j2)
            )

            # Best learned action
            a_idx = choose_best_action(s_key)
            tau1, tau2, tau3, tau4 = ACTIONS[a_idx]

            # Apply torques
            data.ctrl[0] = tau1
            data.ctrl[1] = tau2
            data.ctrl[2] = tau3
            data.ctrl[3] = tau4

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

    print("========== TEST END ==========")
    print(f"Max base qvel observed: {max(base_vel_history):.8f} rad/s\n")

# -------------------------
if __name__ == "__main__":
    test_agent()
