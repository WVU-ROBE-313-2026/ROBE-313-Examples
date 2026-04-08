import numpy as np

# Assuming the previous class is saved in 'central_camera.py'
from central_camera import CentralCamera

# 1. Instantiate default camera (f=15mm, pixel=10um)
cam = CentralCamera(focal=0.015, pixel=10e-6)

# 2. Define the test point
P = np.array([0.3, 0.4, 3.0])

# 3. Project
uv = cam.project(P)

print(f"Projected Pixel: u={uv[0]:.1f}, v={uv[1]:.1f}")
# Output: Projected Pixel: u=790.0, v=712.0

