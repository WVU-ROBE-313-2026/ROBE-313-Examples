"""
distance_from_camera.py
In-Class Exercise: Compute 3D coordinates of P1 and P2
Pure NumPy solution
"""

import numpy as np

# Camera intrinsics
f = 1500.0
u0, v0 = 640.0, 360.0
K = np.array([[f,  0, u0],
              [0,  f, v0],
              [0,  0,  1]])

# Image points (homogeneous)
p1 = np.array([100, 360, 1])
p2 = np.array([500, 360, 1])

# Physical segment length
L = 1.0  # meters

# Disparity in x (pixels)
dx = p2[0] - p1[0]  # 400 px

# Depth Z from similar triangles
Z = (f * L) / dx
print(f"Depth Z = {Z:.2f}")

# Inverse intrinsics
K_inv = np.linalg.inv(K)

# Back-project to 3D
P1 = Z * (K_inv @ p1)
P2 = Z * (K_inv @ p2)

print(f"P1 = [{P1[0]:.2f}, {P1[1]:.2f}, {P1[2]:.2f}]")
print(f"P2 = [{P2[0]:.2f}, {P2[1]:.2f}, {P2[2]:.2f}]")
print(f"Segment length = {np.linalg.norm(P2 - P1):.4f}")
