import time
import numpy as np
import mujoco
import mujoco.viewer

# ---- Configuration ----
URDF_PATH = "simple.urdf" # Use your MJCF-equivalent of simple.urdf
TIMESTEP_HZ = 240.0
IMPULSE_NEWTONS = 10.0           # magnitude of force along local X
OFFSET_LOCAL = np.array([0.0, 0.05, 0.0])  # 5 cm along local Y
FORCE_LOCAL = np.array([IMPULSE_NEWTONS, 0.0, 0.0])

# ---- Load model/data ----
m = mujoco.MjModel.from_xml_path(URDF_PATH)
d = mujoco.MjData(m)

# Set gravity to match bullet example
m.opt.gravity[:] = [0.0, 0.0, -9.81]

# Use the viewer in passive mode so we control stepping
# We'll use the viewer's built-in selection (left double-click selects a body).
# Then press 'f' to apply a one-shot impulse-like force at an offset.
impulse_request = {"bodyid": -1, "do": False}

def key_callback(keycode):
    # Press 'f' to apply the impulse to the currently selected body
    c = chr(keycode) if 32 <= keycode < 127 else ""
    if c.lower() == 'f':
        # viewer.pert.select holds the currently selected body id (-1 if none)
        # We'll set a request flag to apply force next step (while we hold the lock).
        impulse_request["do"] = True

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as v:
    # Configure camera similar to PyBullet view
    with v.lock():
        v.cam.azimuth = 45.0
        v.cam.elevation = -30.0
        v.cam.distance = 1.5
        v.cam.lookat[:] = [0.0, 0.0, 0.2]

    # Run for 30 seconds unless window is closed
    start = time.time()
    dt = 1.0 / TIMESTEP_HZ

    while v.is_running() and (time.time() - start) < 30.0:
        step_start = time.time()

        # If user requested an impulse this step, resolve selection and apply
        if impulse_request["do"]:
            with v.lock():
                # Read selected body id from viewer perturbation state
                sel_body = v.pert.select  # -1 if nothing is selected
                if sel_body is not None and sel_body >= 0:
                    # Transform local force and offset to world frame
                    # Body world pose:
                    xmat = d.xmat[sel_body].reshape(3, 3)  # body rotation world-from-local
                    xpos = d.xpos[sel_body].copy()         # body COM position (world)

                    force_world = xmat @ FORCE_LOCAL
                    offset_world = xmat @ OFFSET_LOCAL

                    # Apply Cartesian force/torque at the body for one step via xfrc_applied:
                    # torque = r x F (world)
                    torque_world = np.cross(offset_world, force_world)

                    # Clear any previous external forces
                    d.xfrc_applied[:] = 0.0

                    # Write 6D wrench (force, torque) at this body’s COM frame (world coordinates)
                    d.xfrc_applied[sel_body, 0:3] = force_world
                    d.xfrc_applied[sel_body, 3:6] = torque_world

                    # Mark we’ve consumed the impulse request
                    impulse_request["do"] = False

        # Step physics
        mujoco.mj_step(m, d)

        # Clear external forces so it acts like a one-step impulse
        d.xfrc_applied[:] = 0.0

        # Sync viewer (also transfers UI state like selection into data/opts)
        v.sync()

        # Simple time-keeping to approximate fixed rate
        sleep_left = dt - (time.time() - step_start)
        if sleep_left > 0:
            time.sleep(sleep_left)
