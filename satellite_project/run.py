import mujoco
import mujoco.viewer
import numpy as np
import time

# Path to the converted MJCF (from URDF)
XML = "satellite_5dof.xml"

# Load model and data
try:
    model = mujoco.MjModel.from_xml_path(XML)
except Exception as e:
    print(f"Error loading '{XML}': {e}")
    print("Convert the URDF first, e.g.: urdf2mjcf satellite_5dof.urdf --output satellite_5dof.xml")
    raise SystemExit(1)

data = mujoco.MjData(model)

# --- Helpers ---
def q_by_name(name):
    """Get joint position (angle) by joint name."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        return None
    q_index = model.jnt_qposadr[jid]
    return data.qpos[q_index]

def dq_by_name(name):
    """Get joint velocity by joint name."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        return None
    dq_index = model.jnt_dofadr[jid]
    return data.qvel[dq_index]

# --- Define your joints ---
joint_names = ["base_joint", "joint1", "joint2", "joint3", "joint4"]

# Last printed positions (for change detection)
last_q = np.full(len(joint_names), np.nan)

print("Starting MuJoCo viewer...\n")
print("Use the sliders in the viewer to manually apply torques to each joint.")

# --- Simulation Loop ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Step physics
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # Read joint angles and velocities
        qs, dqs = [], []
        for name in joint_names:
            q = q_by_name(name)
            dq = dq_by_name(name)
            qs.append(0.0 if q is None else float(q))
            dqs.append(0.0 if dq is None else float(dq))

        qs, dqs = np.array(qs), np.array(dqs)

        # Print if joint angles changed noticeably
        if not np.allclose(qs, last_q, atol=1e-3):
            degs = np.rad2deg(qs)
            vels = np.rad2deg(dqs)
            s = ", ".join([f"{n}={deg:.2f}° ({vel:.2f}°/s)" for n, deg, vel in zip(joint_names, degs, vels)])
            print(f"[{time.strftime('%H:%M:%S')}] {s}")
            last_q = qs.copy()

        # Sync with viewer
        viewer.sync()

