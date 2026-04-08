import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert to grayscale
img = cv2.imread('building2-1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect top 100 corners (Shi-Tomasi method by default)
# params: image, maxCorners, qualityLevel, minDistance
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10, useHarrisDetector=True)
corners = np.int32(corners)

# Draw green circles at each corner location
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

# Display the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Top 100 Corners Detected'); plt.axis('off'); plt.show()