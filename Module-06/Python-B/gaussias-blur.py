import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
gray = cv2.imread('street.png', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(gray, (15,15), 0)

# Display results
plt.subplot(121); plt.imshow(gray, cmap='gray'); plt.title('Original')
plt.subplot(122); plt.imshow(blurred, cmap='gray'); plt.title('Gaussian Blur(15x15)')
plt.show()
