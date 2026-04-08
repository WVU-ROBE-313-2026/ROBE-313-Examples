#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np

# --- Load camera calibration ---
# Loading the specific file generated from the Part 1 exercise
try:
    with np.load("calibration_result_20260330_222922.npz") as X: 
         mtx, dist = X["K"], X["dist"] # Note: Usually saved as 'mtx' and 'dist' in OpenCV tutorials, verify your save keys!
except FileNotFoundError:
    print("Warning: Calibration file not found. Please run the calibration script first.")
    # Fallback to a dummy matrix so the script doesn't crash on NameError
    mtx = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((5, 1), dtype=np.float32)
    print("Using dummy calibration metrics.")

# --- Select ArUco dictionary and create detector ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# --- Capture video from default camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")

print("Press 'q' to quit")

# Known physical size of the marker in meters
markerLength = 0.05  # 5 cm

# Define 3D coordinates of the marker corners in its own frame (for solvePnP)
obj_points = np.array([[-markerLength/2,  markerLength/2, 0],
                       [ markerLength/2,  markerLength/2, 0],
                       [ markerLength/2, -markerLength/2, 0],
                       [-markerLength/2, -markerLength/2, 0]], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        # Draw detected marker boundaries
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Loop through each detected marker
        for i in range(len(ids)):
            corner = corners[i]
            marker_id = ids[i][0]

            # Estimate pose using the modern solvePnP approach
            success, rvec, tvec = cv2.solvePnP(obj_points, corner, mtx, dist)

            if success:
                # Draw the 3D axis on the marker
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)

                # --- EXERCISE SOLUTION ---
                # Format the translation vector (x, y, z in meters)
                tvec_str = f"x:{tvec[0][0]:.2f} y:{tvec[1][0]:.2f} z:{tvec[2][0]:.2f}m"
                
                # Get the top-left corner of the current marker to anchor the text
                text_pos = (int(corner[0][0][0]), int(corner[0][0][1]) - 10)
                
                # Overlay the coordinates on the live feed
                cv2.putText(frame, tvec_str, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

                # Terminal debug output
                print(f"ID: {marker_id} | tvec (m): {tvec.flatten()} | rvec (rad): {rvec.flatten()}")

    cv2.imshow("ArUco Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
