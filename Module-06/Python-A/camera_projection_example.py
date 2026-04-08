import numpy as np

# 1. Define Camera Matrix (K) directly
# f_x = f/rho = 1500, c_x = 640, c_y = 512
K = np.array([[1500, 0, 640],
[ 0, 1500, 512],
[ 0, 0, 1]])

# 2. Define World Point (30cm, 40cm, 3m)
P_world = np.array([0.3, 0.4, 3.0])

# 3. Project (Linear part)
p_hom = K @ P_world # [2370, 2136, 3.0]

# 4. Perspective Divide (Non-linear part)
uv = p_hom[:2] / p_hom[2]
print(f"Pixel Coordinates: {uv}")
# Output: [790. 712.] -> Matches our manual calculation
