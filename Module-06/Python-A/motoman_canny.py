import cv2
img = cv2.imread('Motoman-Robot.png', cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 100, 200)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.imshow('edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

