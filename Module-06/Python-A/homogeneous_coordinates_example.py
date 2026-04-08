import numpy as np

# Camera Parameters
f = 0.015  # 15mm focal length

# 3D World Point (X=0.3m, Y=0.4m, Z=3.0m)
P = np.array([0.3, 0.4, 3.0])

# Homogeneous Projection (Linear)
p_tilde = np.array([f*P[0], f*P[1], P[2]])
print(f"Homogeneous: {p_tilde}")
# Output: [0.0045, 0.0060, 3.0]

# Normalize to get 2D Pixel Coords (Non-Linear)
p = p_tilde[:2] / p_tilde[2]
print(f"Image Plane: {p}")
# Output: [0.0015, 0.0020] -> (1.5mm, 2.0mm)

