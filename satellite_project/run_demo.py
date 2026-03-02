import mujoco
import mujoco.viewer
import time
import numpy as np

# Load the model from the URDF file
model = mujoco.MjModel.from_xml_path('demo.urdf')
data = mujoco.MjData(model)

# Print detailed joint and actuator information
print("=== Joint Information ===")
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_type = model.jnt_type[i]
    joint_axis = model.jnt_axis[i]
    print(f"Joint {i}: {joint_name}, Type: {joint_type}, Axis: {joint_axis}")

print("\n=== Actuator Information ===")
for i in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    joint_id = model.actuator_trnid[i, 0]  # Joint ID that this actuator controls
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    print(f"Control {i}: Actuator '{actuator_name}' -> Joint '{joint_name}' (ID: {joint_id})")

print(f"\nTotal controls available: {model.nu}")
print(f"Control range: {model.actuator_ctrlrange}")

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Configure simulation settings
    model.opt.gravity = [0, 0, -9.81]  # Keep some gravity for stability

    # Initialize control inputs to zero
    data.ctrl[:] = 0

    # Set initial joint positions (optional - for better starting pose)
    if model.nq > 0:
        data.qpos[:] = 0  # All joints start at 0 degrees

    # Set camera properties for view similar to PyBullet's resetDebugVisualizerCamera
    viewer.cam.azimuth = 45          # Horizontal rotation (yaw)
    viewer.cam.elevation = -30     # Vertical rotation (pitch)
    viewer.cam.distance = 1.5
    viewer.cam.lookat[:] = [0, 0, 0.2]

    print("\nUse the sliders in the right panel to control joint angles!")
    print("The sliders should now correspond to the actuators listed above.")

    while viewer.is_running():
        step_start = time.time()

        # The viewer automatically updates data.ctrl based on slider values
        # You can also programmatically set joint angles here if needed:
        # data.ctrl[0] = desired_angle_for_first_joint  # in radians

        # Step the simulation
        mujoco.mj_step(model, data)

        # Sync with viewer
        viewer.sync()

        # Maintain timestep
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

