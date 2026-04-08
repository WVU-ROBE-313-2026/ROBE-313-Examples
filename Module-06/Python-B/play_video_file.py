import cv2
cap = cv2.VideoCapture('tree.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Define circle parameters
    radius = 10  # Adjust as needed
    color = (0, 255, 0)  # Green color in BGR format
    thickness = -1  # Thickness of the circle outline, -1 for a filled circle
    center_coordinates = (20, 15)
    # Draw the circle
    cv2.circle(frame, center_coordinates, radius, color, thickness)

    cv2.imshow('Traffic', frame)

    if cv2.waitKey(100) == ord('q'): # ~30 FPS
        break

cap.release()
cv2.destroyAllWindows()
