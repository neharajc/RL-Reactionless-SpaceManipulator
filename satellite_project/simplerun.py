import pybullet as p
import pybullet_data
import time

# Connect to PyBullet GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For default data paths

# Load ground plane
planeId = p.loadURDF("plane.urdf")

# Load your 2-DOF robot URDF (make sure the file is in the working directory)
robot = p.loadURDF("simple.urdf", [0, 0, 0.1], useFixedBase=False)

# Set gravity
p.setGravity(0, 0, 0)

# Set camera view for better visualization
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.2]
)

# Simulation loop
while True:
    p.stepSimulation()
    time.sleep(1. / 240.)

    mouse_events = p.getMouseEvents()
    for e in mouse_events:
        # Left mouse button down event (button 2, state 3)
        if e[0] == 2 and e[1] == 3:
            cam_info = p.getDebugVisualizerCamera()
            width, height = cam_info[0], cam_info[1]
            mouseX, mouseY = e[2], e[3]

            # Get camera position and forward vector
            cam_pos = cam_info[11]
            cam_target = cam_info[12]
            forward_vec = [cam_target[i] - cam_pos[i] for i in range(3)]

            ray_len = 10
            ray_start = cam_pos
            ray_end = [cam_pos[i] + ray_len * forward_vec[i] for i in range(3)]

            # Perform ray test to detect which link was clicked
            hits = p.rayTest(ray_start, ray_end)
            for hit in hits:
                hitObjUid = hit[0]
                if hitObjUid == robot:
                    link_index = hit[1]
                    hitPos = hit[3]
                    print(f"Clicked link {link_index} at {hitPos}")

                    # Apply an off-center impulse to cause spinning
                    p.applyExternalImpulse(
                        objectUniqueId=robot,
                        linkIndex=link_index,
                        impulse=[10, 0, 0],     # Impulse along the link's local X axis
                        posObj=[0, 0.05, 0],    # 5cm offset along link's local Y to create torque around Z
                        flags=p.LINK_FRAME
                    )

