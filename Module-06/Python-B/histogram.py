import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('flowers8.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the image
plt.imshow(gray, cmap='gray')
plt.title('Street Scene (Grayscale)')
plt.axis('off')
plt.show()

# Display the histogram
plt.hist(gray.ravel(), bins=256, range=(0,256), color='gray')
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Count')
plt.show()
