import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
gray = cv2.imread('street.png', cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)

eroded = cv2.erode(binary, kernel)
dilated = cv2.dilate(binary, kernel)

plt.subplot(131); plt.imshow(binary, cmap='gray'); plt.title('Binary')
plt.subplot(132); plt.imshow(eroded, cmap='gray'); plt.title('Eroded')
plt.subplot(133); plt.imshow(dilated, cmap='gray'); plt.title('Dilated')
plt.show()
