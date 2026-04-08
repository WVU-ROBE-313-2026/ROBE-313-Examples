import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread('street.png', cv2.IMREAD_GRAYSCALE)


gamma = 0.5
img_float = img.astype(float) / 255.0
adj = np.power(img_float, gamma) # I**gamma
adj = (adj * 255).astype('uint8')

# Display results
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
adj_rgb = cv2.cvtColor(adj, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,5))
plt.subplot(121); plt.imshow(img_rgb); plt.title('Original')
plt.subplot(122); plt.imshow(adj_rgb); plt.title('gamma=0.5')
plt.show()

# cv2.waitKey(0)

plt.figure(figsize=(10,5))
plt.subplot(121); plt.hist(img.ravel(), bins=256, range=(0,256), color='gray'); plt.title('Original')
plt.subplot(122); plt.hist(adj.ravel(), bins=256, range=(0,256), color='gray'); plt.title('gamma=0.5')
plt.show()
