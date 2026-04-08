import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('flowers8.png')

resized = cv2.resize(img, (320, 240))
(h, w) = img.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
plt.subplot(131); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Original')
plt.subplot(132); plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)); plt.title('Resized')
plt.subplot(133); plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)); plt.title('Rotated 45')
plt.show()
