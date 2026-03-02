import mujoco as mj
import mujoco.viewer

MODEL_PATH = "simple.xml"

# Load model and data
m = mj.MjModel.from_xml_path(MODEL_PATH)
d = mj.MjData(m)

# Launch viewer in passive (non-blocking) mode
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mj.mj_step(m, d)
        viewer.sync()

