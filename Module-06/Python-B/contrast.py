import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread('street.png', cv2.IMREAD_GRAYSCALE)

alpha = 1.2 # contrast >1 -> stretch
beta = 0 # brightness >0 -> shift up
adj = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Display results
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
adj_rgb = cv2.cvtColor(adj, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,5))
plt.subplot(121); plt.imshow(img_rgb); plt.title('Original')
plt.subplot(122); plt.imshow(adj_rgb); plt.title('alpha=1.5, beta=30')
plt.show()

# cv2.waitKey(0)

plt.figure(figsize=(10,5))
plt.subplot(121); plt.hist(img.ravel(), bins=256, range=(0,256), color='gray'); plt.title('Original')
plt.subplot(122); plt.hist(adj.ravel(), bins=256, range=(0,256), color='gray'); plt.title('alpha=1.5, beta=30')
plt.show()
