import mujoco
import mujoco.viewer

# Path to your MJCF file
xml_path = "satellite.xml"

# Load the model and create data
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Launch interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation started. Close the window to exit.")
    
    # Run until the viewer is closed
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

