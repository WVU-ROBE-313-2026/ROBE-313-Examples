import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('flowers8.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

alpha = 1.5 # contrast
beta = 30 # brightness

adj = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
adj_rgb = cv2.cvtColor(adj, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))
plt.subplot(121); plt.imshow(img_rgb); plt.title('Original')
plt.subplot(122); plt.imshow(adj_rgb); plt.title('Adjusted')
plt.show()

