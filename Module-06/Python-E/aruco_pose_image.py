import cv2
import cv2.aruco as aruco
import numpy as np

# --- Load camera calibration ---
# Attempt to load the calibration file generated from the exercise
try:
    with np.load("calibration_result_20260330_222922.npz") as X:
        mtx, dist = X["K"], X["dist"]
except FileNotFoundError:
    print("Warning: Calibration file not found. Please run the calibration script first.")
    # Fallback to a dummy matrix so the script doesn't crash on NameError
    mtx = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((5, 1), dtype=np.float32)
    print("Using dummy calibration metrics.")

# Load the static image
frame = cv2.imread("scene_with_marker.jpg")
if frame is None:
    print("Error: Could not read 'scene_with_marker.jpg'. Ensure the file is in the correct directory.")
    exit()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Detect markers
corners, ids, rejected = detector.detectMarkers(frame)

# Draw 2D boundaries and print IDs
if ids is not None:
    aruco.drawDetectedMarkers(frame, corners, ids)
    print("Detected IDs:", ids.flatten())

# Show the 2D detection first (optional)
# cv2.imshow("2D Detection", frame)
# cv2.waitKey(0)

# --- 3D Pose Estimation ---
markerLength = 0.05 # 5 cm

# Define 3D coordinates of the marker corners in its own frame (for solvePnP)
obj_points = np.array([[-markerLength/2,  markerLength/2, 0],
                       [ markerLength/2,  markerLength/2, 0],
                       [ markerLength/2, -markerLength/2, 0],
                       [-markerLength/2, -markerLength/2, 0]], dtype=np.float32)

if ids is not None:
    for i in range(len(ids)):
        corner = corners[i]
        
        # Estimate pose using the modern solvePnP approach
        success, rvec, tvec = cv2.solvePnP(obj_points, corner, mtx, dist)
        
        if success:
            # Use the updated OpenCV drawing function
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)

cv2.imshow("Detected Markers with 3D Pose", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
