import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('scene00001.png', 0)
img2 = cv2.imread('scene00004.png', 0)

diff = cv2.absdiff(img1, img2)
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

cv2.imshow('scene00001.png', img1)
cv2.imshow('scene00004.png', img2)
cv2.imshow('diff', thresh)
cv2.waitKey(0)

# plt.imshow(thresh, cmap='gray')
# plt.title('Motion (Binary)')
# plt.show()
