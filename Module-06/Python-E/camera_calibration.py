import cv2
import numpy as np
import os
from datetime import datetime

# ==============================
# CONFIGURATION
# ==============================
CHECKERBOARD = (8, 5)        # Inner corners (width, height)
SQUARE_SIZE = 25.0 / 1000.0  # Size of each square in meters (25 mm)
CAPTURE_KEY = ord(' ')       # Press SPACE to capture a frame
QUIT_KEY = ord('q')          # Press 'q' to quit and calibrate
CAMERA_INDEX = 0             # 0 = default camera (FaceTime HD on Mac)

# Criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points: (0,0,0), (1,0,0), (2,0,0) ... scaled by square size
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Create directory to save captured images (optional)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"calibration_captures_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

print(f"Capturing images will be saved to: {save_dir}")
print(f"Instructions:")
print(f"   - Show checkerboard clearly in different positions/orientations")
print(f"   - Press SPACE to capture a valid frame")
print(f"   - Press 'q' to finish and calibrate")

# ==============================
# OPEN CAMERA
# ==============================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera. Make sure it's not in use by another app.")
    exit()

# Set resolution (optional, helps with performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Verify actual resolution accepted by the hardware
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera initialized at resolution: {actual_w}x{actual_h}")

if actual_w != 1280 or actual_h != 720:
    print("WARNING: Camera did not accept 1280x720. Your K matrix will be scaled to this new resolution!")

    
frame_count = 0
captured_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_copy = frame.copy()

    # Find the chessboard corners
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, draw corners and refine
    if found:
        cv2.drawChessboardCorners(frame_copy, CHECKERBOARD, corners, found)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Show status
        status = f"FOUND! Press SPACE to capture ({captured_count} captured)"
        cv2.putText(frame_copy, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # Capture on spacebar
        key = cv2.waitKey(1) & 0xFF
        if key == CAPTURE_KEY:
            objpoints.append(objp)
            imgpoints.append(corners_refined)

            # Save image
            img_filename = os.path.join(save_dir, f"capture_{captured_count:03d}.jpg")
            cv2.imwrite(img_filename, frame)

            print(f"Captured {captured_count + 1}: {img_filename}")
            captured_count += 1

            # Optional: brief flash effect
            flash = frame_copy.copy()
            cv2.putText(flash, "CAPTURED!", (frame.shape[1]//2 - 150, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
            cv2.imshow('Camera Calibration', flash)
            cv2.waitKey(200)  # Show flash for 200ms
    else:
        cv2.putText(frame_copy, "No checkerboard detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display frame counter and instructions
    info = f"Captured: {captured_count} | Press SPACE to capture | 'q' to calibrate"
    cv2.putText(frame_copy, info, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Camera Calibration', frame_copy)

    # Check for quit
    key = cv2.waitKey(1) & 0xFF
    if key == QUIT_KEY or key == 27:  # 'q' or Esc
        break

cap.release()
cv2.destroyAllWindows()

# ==============================
# CALIBRATION
# ==============================
if len(objpoints) < 5:
    print(f"Warning: Only {len(objpoints)} images captured. Need at least 5–10 for good calibration.")
else:
    print(f"Calibrating with {len(objpoints)} images...")

img_size = gray.shape[::-1]  # (width, height)

print("Running cv2.calibrateCamera...")
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

if ret:
    print("\n" + "="*50)
    print("CALIBRATION SUCCESSFUL!")
    print("="*50)
    print(f"RMSE (Reprojection Error): {ret:.4f} pixels")
    print("\nIntrinsic Matrix K =")
    print(K)
    print("\nDistortion Coefficients [k1, k2, p1, p2, k3] =")
    print(dist.ravel())

    # Save calibration to file
    calib_file = f"calibration_result_{timestamp}.npz"
    np.savez(calib_file,
             K=K,
             dist=dist,
             rvecs=rvecs,
             tvecs=tvecs,
             img_size=img_size)
    print(f"\nCalibration saved to: {calib_file}")

    # Optional: Print extrinsics for each view
    print(f"\nExtrinsic parameters for {len(rvecs)} views:")
    for i in range(len(rvecs)):
        rmat, _ = cv2.Rodrigues(rvecs[i])
        print(f"\n--- View {i+1} ---")
        print(f"Rotation Matrix:\n{rmat}")
        print(f"Translation Vector:\n{tvecs[i].flatten()}")
else:
    print("Calibration failed.")
