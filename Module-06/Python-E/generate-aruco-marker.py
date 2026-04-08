import cv2
import cv2.aruco as aruco

# Select dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Create marker ID 23
marker_img = aruco.generateImageMarker(aruco_dict, 23, 200)

# Save or show
cv2.imwrite("marker23.png", marker_img)
cv2.imshow("Marker", marker_img)
cv2.waitKey(0)
