import pybullet as p
import pybullet_data
import time

# Connect to PyBullet GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load ground plane and robot
planeId = p.loadURDF("plane.urdf")
robot = p.loadURDF("satellite_5dof_pybullet.urdf", [0, 0, 0.1], useFixedBase=False)

# Set gravity
p.setGravity(0, 0, -9.81)

# Set initial camera view
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.2]
)

while True:
    p.stepSimulation()
    time.sleep(1. / 240.)

    mouse_events = p.getMouseEvents()
    for e in mouse_events:
        # Left mouse button down event
        if e[0] == 2 and e[1] == 3:
            cam_info = p.getDebugVisualizerCamera()
            width, height = cam_info[0], cam_info[1]
            viewMat = cam_info[2]
            projMat = cam_info[3]
            mouseX, mouseY = e[2], e[3]

            # Convert 2D mouse coordinates to normalized device coordinates [-1,1]
            norm_x = (mouseX / width) * 2 - 1
            norm_y = -((mouseY / height) * 2 - 1)

            # Get camera position and forward vector
            cam_pos = cam_info[11]
            cam_target = cam_info[12]
            forward_vec = [cam_target[i] - cam_pos[i] for i in range(3)]
            ray_len = 10
            ray_start = cam_pos
            ray_end = [cam_pos[i] + ray_len * forward_vec[i] for i in range(3)]

            # Perform ray test
            hits = p.rayTest(ray_start, ray_end)
            for hit in hits:
                hitObjUid = hit[0]
                if hitObjUid == robot:
                    link_index = hit[1]
                    hitPos = hit[3]
                    print(f"Clicked link {link_index} at {hitPos}")
                    # Apply instant backward impulse along the local negative X axis of the link
                    p.applyExternalImpulse(
                        objectUniqueId=robot,
                        linkIndex=link_index,
                        impulse=[-50, 0, 0],  # backward impulse strength
                        posObj=hitPos,
                        flags=p.LINK_FRAME
                    )
                    # ALSO apply a sideways force with y-offset to create z-axis rotation
                    p.applyExternalForce(
                        objectUniqueId=robot,
                        linkIndex=link_index,
                        forceObj=[20, 0, 0],          # force along link X
                        posObj=[0, 0.05, 0],          # offset +5cm along link Y to create torque around Z
                        flags=p.LINK_FRAME
                    )

