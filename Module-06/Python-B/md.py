import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit("Camera failed")

# MOG2 keeps a rolling history of the background
bg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret: break

    # Extract moving foreground (white) from static background (black)
    mask = bg.apply(frame)
    
    # MORPH_OPEN = Erosion followed by Dilation (Removes noise!)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Color the moving pixels red
    result = frame.copy()
    result[mask == 255] = [0, 0, 255]  # BGR format

    cv2.imshow('Motion (q=quit)', result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()

