import numpy as np

# 1. World Point (Homogeneous)
P_world = np.array([0.3, 0.4, 3.0, 1])

# 2. Camera Pose (Shifted x = -0.5m)
# Camera is at -0.5m looking forward
T_pose = np.eye(4)
T_pose[0, 3] = -0.5

# 3. Extrinsics = Inverse(Pose)
E = np.linalg.inv(T_pose)

# 4. Transform & Project
P_cam = E @ P_world # [0.8, 0.4, 3.0, 1]
x = 0.015 * P_cam[0] / P_cam[2] # 0.0040
y = 0.015 * P_cam[1] / P_cam[2] # 0.0020
