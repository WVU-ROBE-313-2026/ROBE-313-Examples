import cv2

cap = cv2.VideoCapture(0) # 0 is usually the default webcam

# Always check if the hardware initialized properly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Failed to grab frame")
        break

    cv2.imshow('Live Feed - Press q to quit', frame)
    
    # Wait 1ms for GUI event. Mask with 0xFF for cross-platform safety.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

